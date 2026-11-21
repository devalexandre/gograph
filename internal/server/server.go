// Package server expõe uma interface web (React + Tailwind) para navegar no
// grafo de chamadas e visualizar o código-fonte das funções indexadas.
package server

import (
	"database/sql"
	"embed"
	"encoding/json"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/devalexandre/gograph/internal/store"
)

//go:embed ui/index.html
var uiFS embed.FS

type Node struct {
	ID          int64  `json:"id"`
	Name        string `json:"name"`
	Package     string `json:"package"`
	File        string `json:"file"`
	LineStart   int    `json:"lineStart"`
	LineEnd     int    `json:"lineEnd"`
	Signature   string `json:"signature"`
	External    bool   `json:"external"`
	Description string `json:"description,omitempty"`
}

type Edge struct {
	Caller int64  `json:"caller"`
	Callee int64  `json:"callee"`
	Kind   string `json:"kind"`
}

type GraphPayload struct {
	Module string `json:"module"`
	Root   string `json:"root"`
	Nodes  []Node `json:"nodes"`
	Edges  []Edge `json:"edges"`
}

type Ref struct {
	ID      int64  `json:"id"`
	Name    string `json:"name"`
	Package string `json:"package"`
	Kind    string `json:"kind"`
}

type SourcePayload struct {
	Node
	Code     string `json:"code"`
	Calls    []Ref  `json:"calls"`
	CalledBy []Ref  `json:"calledBy"`
}

// Handler devolve o http.Handler da interface web e das APIs do grafo.
func Handler(st *store.Store, root string) (http.Handler, error) {
	sub, err := fs.Sub(uiFS, "ui")
	if err != nil {
		return nil, err
	}
	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.FS(sub)))
	mux.HandleFunc("/api/graph", func(w http.ResponseWriter, r *http.Request) {
		payload, err := loadGraph(st, root)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, payload)
	})
	mux.HandleFunc("/api/source", func(w http.ResponseWriter, r *http.Request) {
		id, err := strconv.ParseInt(r.URL.Query().Get("id"), 10, 64)
		if err != nil || id <= 0 {
			http.Error(w, "id inválido", http.StatusBadRequest)
			return
		}
		payload, err := loadSource(st, root, id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		writeJSON(w, payload)
	})
	return mux, nil
}

func loadGraph(st *store.Store, root string) (*GraphPayload, error) {
	db := st.DB()
	var mod, savedRoot string
	_ = db.QueryRow(`SELECT value FROM meta WHERE key = 'module'`).Scan(&mod)
	_ = db.QueryRow(`SELECT value FROM meta WHERE key = 'root'`).Scan(&savedRoot)
	if root == "" {
		root = savedRoot
	}

	rows, err := db.Query(`
		SELECT id, name, package, file, line_start, line_end, signature, external, COALESCE(description, '')
		FROM functions
		ORDER BY external, package, file, line_start, name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var nodes []Node
	for rows.Next() {
		var n Node
		var ext int
		if err := rows.Scan(&n.ID, &n.Name, &n.Package, &n.File, &n.LineStart, &n.LineEnd, &n.Signature, &ext, &n.Description); err != nil {
			return nil, err
		}
		n.External = ext != 0
		nodes = append(nodes, n)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	erows, err := db.Query(`SELECT caller_id, callee_id, kind FROM edges`)
	if err != nil {
		return nil, err
	}
	defer erows.Close()

	var edges []Edge
	for erows.Next() {
		var e Edge
		if err := erows.Scan(&e.Caller, &e.Callee, &e.Kind); err != nil {
			return nil, err
		}
		edges = append(edges, e)
	}
	if err := erows.Err(); err != nil {
		return nil, err
	}

	return &GraphPayload{Module: mod, Root: root, Nodes: nodes, Edges: edges}, nil
}

func loadSource(st *store.Store, root string, id int64) (*SourcePayload, error) {
	db := st.DB()
	if root == "" {
		_ = db.QueryRow(`SELECT value FROM meta WHERE key = 'root'`).Scan(&root)
	}
	var out SourcePayload
	var ext int
	err := db.QueryRow(`
		SELECT id, name, package, file, line_start, line_end, signature, external, COALESCE(description, '')
		FROM functions WHERE id = ?`, id).Scan(
		&out.ID, &out.Name, &out.Package, &out.File, &out.LineStart, &out.LineEnd, &out.Signature, &ext, &out.Description,
	)
	if err != nil {
		return nil, err
	}
	out.External = ext != 0

	if !out.External && out.File != "" && out.LineStart > 0 && root != "" {
		if data, err := os.ReadFile(filepath.Join(root, out.File)); err == nil {
			lines := strings.Split(string(data), "\n")
			from := out.LineStart - 1
			to := out.LineEnd
			if from < 0 {
				from = 0
			}
			if to > len(lines) {
				to = len(lines)
			}
			if from < to {
				out.Code = strings.Join(lines[from:to], "\n")
			}
		}
	}

	out.Calls, _ = queryRefs(db, `
		SELECT f.id, f.name, f.package, e.kind
		FROM edges e JOIN functions f ON f.id = e.callee_id
		WHERE e.caller_id = ?
		ORDER BY f.external, f.name`, id)
	out.CalledBy, _ = queryRefs(db, `
		SELECT f.id, f.name, f.package, e.kind
		FROM edges e JOIN functions f ON f.id = e.caller_id
		WHERE e.callee_id = ?
		ORDER BY f.name`, id)
	return &out, nil
}

func queryRefs(db *sql.DB, q string, id int64) ([]Ref, error) {
	rows, err := db.Query(q, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Ref
	for rows.Next() {
		var r Ref
		if err := rows.Scan(&r.ID, &r.Name, &r.Package, &r.Kind); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}
