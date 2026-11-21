package describe

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/devalexandre/gograph/internal/store"
)

type fakeBackend struct {
	mu    sync.Mutex
	calls int
}

func (f *fakeBackend) Name() string { return "fake" }
func (f *fakeBackend) Generate(ctx context.Context, system, user string) (string, error) {
	f.mu.Lock()
	f.calls++
	f.mu.Unlock()
	return `{"purpose":"faz algo","calls":[],"effects":[],"errors":[],"concurrency":""}`, nil
}

// Grafo com cadeia (a -> b -> c) e ciclo (x <-> y): tudo precisa ser descrito
// uma única vez, mesmo com vários workers.
func TestRunDescribesEverythingOnce(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "f.go"), []byte("package p\n\nfunc f() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	st, err := store.Open(filepath.Join(root, "g.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	var funcs []store.Function
	for _, n := range []string{"a", "b", "c", "x", "y"} {
		funcs = append(funcs, store.Function{Name: "p." + n, Package: "p", File: "f.go", LineStart: 3, LineEnd: 3, Hash: "h" + n})
	}
	edges := []store.Edge{
		{Caller: "p.a", Callee: "p.b", Kind: store.KindStatic},
		{Caller: "p.b", Callee: "p.c", Kind: store.KindStatic},
		{Caller: "p.x", Callee: "p.y", Kind: store.KindStatic},
		{Caller: "p.y", Callee: "p.x", Kind: store.KindStatic},
	}
	if _, err := st.ReplaceGraph(funcs, edges, map[string]string{}); err != nil {
		t.Fatal(err)
	}

	be := &fakeBackend{}
	var order []string
	rep, err := Run(context.Background(), st, Options{Root: root, Backend: be, Workers: 3,
		Progress: func(done, total int, name string, err error) { order = append(order, name) }})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Described != 5 || rep.Failed != 0 || be.calls != 5 {
		t.Fatalf("descritas=%d falhas=%d chamadas=%d", rep.Described, rep.Failed, be.calls)
	}
	pos := map[string]int{}
	for i, n := range order {
		pos[n] = i
	}
	if !(pos["p.c"] < pos["p.b"] && pos["p.b"] < pos["p.a"]) {
		t.Errorf("ordem deveria ser c, b, a: %v", order)
	}
	if pending, _ := st.PendingDescriptions(); len(pending) != 0 {
		t.Errorf("sobraram %d pendentes", len(pending))
	}
}
