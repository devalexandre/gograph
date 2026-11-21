// Package store persiste o grafo de funções em SQLite.
package store

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// SchemaVersion muda sempre que o esquema das tabelas muda.
const SchemaVersion = "1"

// Function é um nó do grafo. Nós externos (stdlib e dependências) têm
// External=true e ficam sem hash e sem descrição.
type Function struct {
	Name      string
	Package   string
	File      string
	LineStart int
	LineEnd   int
	Signature string
	Hash      string
	External  bool
}

// Edge é uma chamada de Caller para Callee, identificados pelo nome completo.
type Edge struct {
	Caller string
	Callee string
	Kind   string
}

// Tipos de aresta.
const (
	KindStatic    = "static"    // chamada direta a uma função conhecida
	KindInterface = "interface" // chamada via método de interface
	KindClosure   = "closure"   // função anônima definida ou chamada pelo pai
	KindDynamic   = "dynamic"   // chamada via valor do tipo func
	KindGo        = "go"        // iniciada com go
	KindDefer     = "defer"     // adiada com defer
)

const schema = `
CREATE TABLE IF NOT EXISTS functions (
	id          INTEGER PRIMARY KEY,
	name        TEXT NOT NULL UNIQUE,
	package     TEXT NOT NULL,
	file        TEXT NOT NULL DEFAULT '',
	line_start  INTEGER NOT NULL DEFAULT 0,
	line_end    INTEGER NOT NULL DEFAULT 0,
	signature   TEXT NOT NULL DEFAULT '',
	hash        TEXT NOT NULL DEFAULT '',
	external    INTEGER NOT NULL DEFAULT 0,
	description TEXT,
	desc_hash   TEXT,
	updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_functions_package ON functions(package);
CREATE INDEX IF NOT EXISTS idx_functions_hash ON functions(hash);

CREATE TABLE IF NOT EXISTS edges (
	caller_id INTEGER NOT NULL REFERENCES functions(id) ON DELETE CASCADE,
	callee_id INTEGER NOT NULL REFERENCES functions(id) ON DELETE CASCADE,
	kind      TEXT NOT NULL,
	PRIMARY KEY (caller_id, callee_id, kind)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_edges_callee ON edges(callee_id);

CREATE TABLE IF NOT EXISTS meta (
	key   TEXT PRIMARY KEY,
	value TEXT NOT NULL
) WITHOUT ROWID;
`

type Store struct {
	db *sql.DB
}

// Open abre (ou cria) o banco em path.
func Open(path string) (*Store, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite3", path+"?_foreign_keys=on&_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return nil, err
	}
	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("criar esquema: %w", err)
	}
	return &Store{db: db}, nil
}

func (s *Store) Close() error { return s.db.Close() }

// DB expõe a conexão para consultas de leitura.
func (s *Store) DB() *sql.DB { return s.db }

// Stats resume o que mudou em um ReplaceGraph.
type Stats struct {
	Functions int // nós do módulo
	External  int // nós externos
	Edges     int
	Added     int // funções do módulo que não existiam
	Changed   int // funções do módulo com hash diferente
	Removed   int
}

// ReplaceGraph grava o grafo completo em uma transação. Funções que já
// existiam mantêm id, description e desc_hash; a descrição fica obsoleta
// quando desc_hash difere de hash.
func (s *Store) ReplaceGraph(funcs []Function, edges []Edge, meta map[string]string) (Stats, error) {
	var st Stats
	tx, err := s.db.Begin()
	if err != nil {
		return st, err
	}
	defer tx.Rollback()

	old := map[string]string{} // nome -> hash
	rows, err := tx.Query(`SELECT name, hash FROM functions`)
	if err != nil {
		return st, err
	}
	for rows.Next() {
		var name, hash string
		if err := rows.Scan(&name, &hash); err != nil {
			rows.Close()
			return st, err
		}
		old[name] = hash
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return st, err
	}

	now := time.Now().UTC().Format(time.RFC3339)
	upsert, err := tx.Prepare(`
		INSERT INTO functions (name, package, file, line_start, line_end, signature, hash, external, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(name) DO UPDATE SET
			package = excluded.package, file = excluded.file,
			line_start = excluded.line_start, line_end = excluded.line_end,
			signature = excluded.signature, external = excluded.external,
			updated_at = CASE WHEN functions.hash = excluded.hash THEN functions.updated_at ELSE excluded.updated_at END,
			hash = excluded.hash`)
	if err != nil {
		return st, err
	}
	defer upsert.Close()

	seen := make(map[string]bool, len(funcs))
	for _, f := range funcs {
		if seen[f.Name] {
			continue
		}
		seen[f.Name] = true
		if _, err := upsert.Exec(f.Name, f.Package, f.File, f.LineStart, f.LineEnd, f.Signature, f.Hash, f.External, now); err != nil {
			return st, fmt.Errorf("gravar %s: %w", f.Name, err)
		}
		if f.External {
			st.External++
			continue
		}
		st.Functions++
		if h, ok := old[f.Name]; !ok {
			st.Added++
		} else if h != f.Hash {
			st.Changed++
		}
	}

	del, err := tx.Prepare(`DELETE FROM functions WHERE name = ?`)
	if err != nil {
		return st, err
	}
	defer del.Close()
	for name := range old {
		if !seen[name] {
			if _, err := del.Exec(name); err != nil {
				return st, err
			}
			st.Removed++
		}
	}

	ids := make(map[string]int64, len(seen))
	rows, err = tx.Query(`SELECT id, name FROM functions`)
	if err != nil {
		return st, err
	}
	for rows.Next() {
		var id int64
		var name string
		if err := rows.Scan(&id, &name); err != nil {
			rows.Close()
			return st, err
		}
		ids[name] = id
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return st, err
	}

	if _, err := tx.Exec(`DELETE FROM edges`); err != nil {
		return st, err
	}
	ins, err := tx.Prepare(`INSERT OR IGNORE INTO edges (caller_id, callee_id, kind) VALUES (?, ?, ?)`)
	if err != nil {
		return st, err
	}
	defer ins.Close()
	for _, e := range edges {
		from, ok1 := ids[e.Caller]
		to, ok2 := ids[e.Callee]
		if !ok1 || !ok2 {
			continue
		}
		res, err := ins.Exec(from, to, e.Kind)
		if err != nil {
			return st, err
		}
		if n, _ := res.RowsAffected(); n > 0 {
			st.Edges++
		}
	}

	meta["schema_version"] = SchemaVersion
	for k, v := range meta {
		if _, err := tx.Exec(`INSERT INTO meta (key, value) VALUES (?, ?)
			ON CONFLICT(key) DO UPDATE SET value = excluded.value`, k, v); err != nil {
			return st, err
		}
	}
	return st, tx.Commit()
}
