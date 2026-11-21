package implement

import (
	"reflect"
	"testing"
)

func TestCompilerErrors(t *testing.T) {
	out := "# github.com/gorilla/mux\n" +
		"/tmp/x/replaced.go:415:20: not enough arguments in call to walkFn\n" +
		"\thave (*Route, []*Route)\n" +
		"\twant (*Route, *Router, []*Route)\n" +
		"/tmp/x/replaced.go:30:2: \"fmt\" imported and not used\n" +
		"./other.go:9:3: undefined: gone\n"
	got := compilerErrors(out, "replaced.go", 410, 12)
	want := []string{
		"linha 6: not enough arguments in call to walkFn; have (*Route, []*Route); want (*Route, *Router, []*Route)",
		"fora da função: \"fmt\" imported and not used",
		"em other.go: undefined: gone",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %q\nwant %q", got, want)
	}

	syntax := "código inválido: /src/mux/regexp.go:331:40: missing ',' before newline in argument list (and 10 more errors)"
	got = compilerErrors(syntax, "regexp.go", 329, 10)
	if len(got) != 1 || got[0] != "linha 3: missing ',' before newline in argument list (and 10 more errors)" {
		t.Errorf("erro de sintaxe: %q", got)
	}
}

func TestLocate(t *testing.T) {
	src := []byte("package p\n\nimport \"fmt\"\n\ntype T struct{}\n\n// doc\nfunc (t *T) Run(n int) {\n\tfmt.Println(n)\n}\n\nfunc Run() {}\n")
	fn, start := locate(src, "func (t *T) Run(n int) {\n  fmt.Println(n)\n}")
	if start != 8 || fn != "func (t *T) Run(n int) {\n\tfmt.Println(n)\n}" {
		t.Errorf("método: linha %d, texto %q", start, fn)
	}
	if _, start := locate(src, "func Run() {}"); start != 12 {
		t.Errorf("função: linha %d", start)
	}
}
