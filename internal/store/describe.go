package store

import "database/sql"

// FuncRow é uma função do módulo lida do banco.
type FuncRow struct {
	ID          int64
	Name        string
	Package     string
	File        string
	LineStart   int
	LineEnd     int
	Signature   string
	Hash        string
	Description string // JSON da descrição; vazio se ainda não foi gerada
}

// Callee é uma função chamada, vista a partir de quem chama.
type Callee struct {
	ID          int64
	Name        string
	Package     string
	Signature   string
	Description string
	External    bool
	Kind        string
}

// Meta devolve o valor de uma chave da tabela meta, ou vazio.
func (s *Store) Meta(key string) (string, error) {
	var v string
	err := s.db.QueryRow(`SELECT value FROM meta WHERE key = ?`, key).Scan(&v)
	if err == sql.ErrNoRows {
		return "", nil
	}
	return v, err
}

// SetMeta grava uma chave na tabela meta.
func (s *Store) SetMeta(key, value string) error {
	_, err := s.db.Exec(`INSERT INTO meta (key, value) VALUES (?, ?)
		ON CONFLICT(key) DO UPDATE SET value = excluded.value`, key, value)
	return err
}

// PendingDescriptions lista as funções do módulo sem descrição ou com a
// descrição obsoleta (gerada para outro hash).
func (s *Store) PendingDescriptions() ([]FuncRow, error) {
	rows, err := s.db.Query(`
		SELECT id, name, package, file, line_start, line_end, signature, hash
		FROM functions
		WHERE external = 0 AND hash <> '' AND (description IS NULL OR desc_hash IS NOT hash)
		ORDER BY name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []FuncRow
	for rows.Next() {
		var f FuncRow
		if err := rows.Scan(&f.ID, &f.Name, &f.Package, &f.File, &f.LineStart, &f.LineEnd, &f.Signature, &f.Hash); err != nil {
			return nil, err
		}
		out = append(out, f)
	}
	return out, rows.Err()
}

// Callees lista as funções chamadas diretamente por callerID.
func (s *Store) Callees(callerID int64) ([]Callee, error) {
	rows, err := s.db.Query(`
		SELECT c.id, c.name, c.package, c.signature, COALESCE(c.description, ''), c.external, e.kind
		FROM edges e JOIN functions c ON c.id = e.callee_id
		WHERE e.caller_id = ?
		ORDER BY c.external, c.name, e.kind`, callerID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Callee
	for rows.Next() {
		var c Callee
		if err := rows.Scan(&c.ID, &c.Name, &c.Package, &c.Signature, &c.Description, &c.External, &c.Kind); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

// SetDescription grava a descrição gerada para o hash informado.
func (s *Store) SetDescription(id int64, description, hash string) error {
	_, err := s.db.Exec(`UPDATE functions SET description = ?, desc_hash = ? WHERE id = ?`, description, hash, id)
	return err
}
