package implement

import (
	"context"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"io"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"golang.org/x/tools/go/packages"

	"github.com/devalexandre/gograph/internal/store"
)

// Symbol é algo declarado fora da função que o corpo original usa.
type Symbol struct {
	Name   string `json:"name"`
	Kind   string `json:"kind"`   // função, método, campo, tipo, constante, variável
	Origin string `json:"origin"` // pacote, projeto, externo, stdlib
	Owner  string `json:"owner,omitempty"`
	Shown  bool   `json:"shown"` // o nome aparece na entrada do modelo
}

// AuditRow é a auditoria de uma função: o que ela usa e o que a entrada mostra.
type AuditRow struct {
	Name    string   `json:"name"`
	Hash    string   `json:"hash"`
	Symbols []Symbol `json:"symbols"`
	// Missing conta os símbolos do próprio projeto (pacote ou projeto) que o
	// corpo usa e a entrada não mostra: o que o modelo teria de adivinhar.
	Missing int `json:"missing"`
}

// AuditOptions configura a auditoria.
type AuditOptions struct {
	Root               string
	N                  int
	MinLines, MaxLines int
	Tail               bool
	Env                []string
	Out                io.Writer
}

// Audit confere, para as mesmas funções que o medidor usa, se a entrada
// montada por BuildTask mostra o que o corpo original precisa. Usa o
// verificador de tipos do Go: cada nome do corpo é resolvido para a sua
// declaração, então "campo de um tipo do projeto" e "método da biblioteca
// padrão" não se confundem.
func Audit(ctx context.Context, st *store.Store, opts AuditOptions) ([]AuditRow, error) {
	funcs, err := candidates(st, opts.Root, opts.N, opts.MinLines, opts.MaxLines, opts.Tail)
	if err != nil {
		return nil, err
	}
	cfg := &packages.Config{
		Dir: opts.Root, Env: opts.Env, Context: ctx,
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles | packages.NeedImports |
			packages.NeedDeps | packages.NeedTypes | packages.NeedSyntax | packages.NeedTypesInfo | packages.NeedModule,
	}
	pkgs, err := packages.Load(cfg, "./...")
	if err != nil {
		return nil, fmt.Errorf("carregar pacotes: %w", err)
	}
	type located struct {
		pkg  *packages.Package
		file *ast.File
	}
	byFile := map[string]located{}
	module := ""
	for _, p := range pkgs {
		if p.Module != nil && module == "" {
			module = p.Module.Path
		}
		for i, f := range p.Syntax {
			if i < len(p.CompiledGoFiles) {
				byFile[p.CompiledGoFiles[i]] = located{p, f}
			}
		}
	}

	var rows []AuditRow
	for _, f := range funcs {
		loc, ok := byFile[filepath.Join(opts.Root, f.File)]
		if !ok || loc.pkg.TypesInfo == nil {
			continue
		}
		var decl *ast.FuncDecl
		for _, d := range loc.file.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && loc.pkg.Fset.Position(fd.Pos()).Line == f.LineStart {
				decl = fd
			}
		}
		if decl == nil || decl.Body == nil {
			continue
		}
		task, err := BuildTask(opts.Root, st.DB(), f)
		if err != nil {
			continue
		}
		row := AuditRow{Name: f.Name, Hash: f.Hash}
		row.Symbols = usedSymbols(loc.pkg, decl, module, task.Prompt)
		for _, s := range row.Symbols {
			if !s.Shown && (s.Origin == "pacote" || s.Origin == "projeto") {
				row.Missing++
			}
		}
		rows = append(rows, row)
		if opts.Out != nil {
			line, _ := json.Marshal(row)
			fmt.Fprintf(opts.Out, "%s\n", line)
		}
	}
	return rows, nil
}

// usedSymbols lista o que o corpo de decl usa e que foi declarado fora dele.
func usedSymbols(pkg *packages.Package, decl *ast.FuncDecl, module, prompt string) []Symbol {
	info := pkg.TypesInfo
	inside := func(pos token.Pos) bool { return pos >= decl.Pos() && pos <= decl.End() }
	self := info.Defs[decl.Name]
	seen := map[types.Object]bool{}
	out := []Symbol{}
	ast.Inspect(decl.Body, func(n ast.Node) bool {
		id, ok := n.(*ast.Ident)
		if !ok {
			return true
		}
		obj := info.Uses[id]
		if obj == nil || obj == self || seen[obj] || obj.Pkg() == nil || inside(obj.Pos()) {
			return true
		}
		if _, isPkg := obj.(*types.PkgName); isPkg {
			return true
		}
		seen[obj] = true
		s := Symbol{Name: obj.Name(), Origin: origin(obj.Pkg().Path(), pkg.PkgPath, module)}
		switch o := obj.(type) {
		case *types.Func:
			s.Kind = "função"
			if sig, ok := o.Type().(*types.Signature); ok && sig.Recv() != nil {
				s.Kind, s.Owner = "método", typeName(sig.Recv().Type())
			}
		case *types.Var:
			s.Kind = "variável"
			if o.IsField() {
				s.Kind = "campo"
			}
		case *types.TypeName:
			s.Kind = "tipo"
		case *types.Const:
			s.Kind = "constante"
		default:
			return true
		}
		s.Shown = wordIn(prompt, s.Name)
		out = append(out, s)
		return true
	})
	sort.Slice(out, func(i, j int) bool {
		if out[i].Origin != out[j].Origin {
			return out[i].Origin < out[j].Origin
		}
		return out[i].Name < out[j].Name
	})
	return out
}

func origin(path, own, module string) string {
	switch {
	case path == own:
		return "pacote"
	case module != "" && (path == module || strings.HasPrefix(path, module+"/")):
		return "projeto"
	case !strings.Contains(strings.SplitN(path, "/", 2)[0], "."):
		return "stdlib"
	}
	return "externo"
}

func typeName(t types.Type) string {
	if p, ok := t.(*types.Pointer); ok {
		t = p.Elem()
	}
	if n, ok := t.(*types.Named); ok {
		return n.Obj().Name()
	}
	return t.String()
}

func wordIn(text, word string) bool {
	re, err := regexp.Compile(`\b` + regexp.QuoteMeta(word) + `\b`)
	return err == nil && re.MatchString(text)
}
