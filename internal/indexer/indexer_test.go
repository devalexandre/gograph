package indexer

import (
	"path/filepath"
	"testing"

	"github.com/devalexandre/gograph/internal/store"
)

func TestHashIgnoresCommentsAndFormatting(t *testing.T) {
	a := "func f(a int) int {\n\tx := a + 1\n\treturn x\n}"
	b := "func f( a int ) int {\n\t// soma um\n\tx := a+1 /* aqui */\n\n\n\treturn x\n}"
	oneLine := "func f(a int) int { x := a + 1; return x }"
	changed := "func f(a int) int {\n\tx := a + 2\n\treturn x\n}"

	if hashSource([]byte(a)) != hashSource([]byte(b)) {
		t.Error("comentário e formatação mudaram o hash")
	}
	if hashSource([]byte(a)) != hashSource([]byte(oneLine)) {
		t.Error("quebra de linha mudou o hash")
	}
	if hashSource([]byte(a)) == hashSource([]byte(changed)) {
		t.Error("mudança de código não mudou o hash")
	}
}

func TestIndexSample(t *testing.T) {
	res, err := Index(Options{Dir: filepath.Join("testdata", "sample")})
	if err != nil {
		t.Fatal(err)
	}
	const p = "example.com/sample."

	funcs := map[string]store.Function{}
	for _, f := range res.Functions {
		funcs[f.Name] = f
	}
	for _, name := range []string{
		"main", "process", "cleanup", "metrics", "Map", "double", "neverCalled",
		"(*FileRepo).Save", "MemRepo.Save", "Unused.Save", "main$1",
	} {
		f, ok := funcs[p+name]
		if !ok {
			t.Errorf("função ausente: %s", name)
			continue
		}
		if f.External || f.Hash == "" || f.File != "main.go" || f.LineStart == 0 {
			t.Errorf("%s incompleta: %+v", name, f)
		}
	}
	if f := funcs[p+"process"]; f.Signature != "func process(s Saver, v string) error" {
		t.Errorf("assinatura de process: %q", f.Signature)
	}
	if f, ok := funcs["strings.ToUpper"]; !ok || !f.External {
		t.Errorf("strings.ToUpper deveria ser nó externo: %+v", f)
	}

	edges := map[store.Edge]bool{}
	for _, e := range res.Edges {
		edges[e] = true
	}
	want := []store.Edge{
		{Caller: p + "main", Callee: p + "process", Kind: store.KindStatic},
		{Caller: p + "process", Callee: p + "(*FileRepo).Save", Kind: store.KindInterface},
		{Caller: p + "process", Callee: p + "MemRepo.Save", Kind: store.KindInterface},
		{Caller: p + "process", Callee: p + "cleanup", Kind: store.KindDefer},
		{Caller: p + "process", Callee: p + "metrics", Kind: store.KindGo},
		{Caller: p + "main", Callee: p + "Map", Kind: store.KindStatic},
		{Caller: p + "Map", Callee: p + "double", Kind: store.KindDynamic},
		{Caller: p + "main", Callee: p + "main$1", Kind: store.KindClosure},
		{Caller: p + "main$1", Callee: p + "cleanup", Kind: store.KindStatic},
		{Caller: p + "(*FileRepo).Save", Callee: "strings.ToUpper", Kind: store.KindStatic},
		{Caller: p + "report", Callee: "error.Error", Kind: store.KindInterface},
	}
	for _, e := range want {
		if !edges[e] {
			t.Errorf("aresta ausente: %s -[%s]-> %s", e.Caller, e.Kind, e.Callee)
		}
	}
	for e := range edges {
		if e.Caller == p+"report" && e.Callee != "error.Error" {
			t.Errorf("implementação externa de error não deveria virar aresta: %s", e.Callee)
		}
	}
	// O VTA não deve ligar process a um tipo que nunca chega nele.
	if edges[store.Edge{Caller: p + "process", Callee: p + "Unused.Save", Kind: store.KindInterface}] {
		t.Error("aresta imprecisa: process -> Unused.Save")
	}
}

func TestReplaceGraphKeepsDescriptions(t *testing.T) {
	st, err := store.Open(filepath.Join(t.TempDir(), "g.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	funcs := []store.Function{{Name: "a.F", Package: "a", Hash: "h1"}, {Name: "a.G", Package: "a", Hash: "h2"}}
	edges := []store.Edge{{Caller: "a.F", Callee: "a.G", Kind: store.KindStatic}}
	if _, err := st.ReplaceGraph(funcs, edges, map[string]string{}); err != nil {
		t.Fatal(err)
	}
	if _, err := st.DB().Exec(`UPDATE functions SET description = 'd', desc_hash = hash WHERE name = 'a.F'`); err != nil {
		t.Fatal(err)
	}

	funcs = []store.Function{{Name: "a.F", Package: "a", Hash: "h1-novo"}, {Name: "a.H", Package: "a", Hash: "h3"}}
	stats, err := st.ReplaceGraph(funcs, nil, map[string]string{})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Added != 1 || stats.Changed != 1 || stats.Removed != 1 || stats.Edges != 0 {
		t.Errorf("stats inesperado: %+v", stats)
	}
	var desc, descHash, hash string
	if err := st.DB().QueryRow(`SELECT description, desc_hash, hash FROM functions WHERE name = 'a.F'`).Scan(&desc, &descHash, &hash); err != nil {
		t.Fatal(err)
	}
	if desc != "d" || descHash != "h1" || hash != "h1-novo" {
		t.Errorf("descrição não preservada: %q %q %q", desc, descHash, hash)
	}
}
