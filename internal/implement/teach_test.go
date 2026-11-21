package implement

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/devalexandre/gograph/internal/corpus"
	"github.com/devalexandre/gograph/internal/store"
)

// fakeTeacher responde como o professor, a partir da própria função: a
// alteração acrescenta uma instrução inócua no começo e a versão quebrada chama uma
// função que não existe.
type fakeTeacher struct{}

func (fakeTeacher) Name() string { return "fake" }

func (fakeTeacher) Generate(_ context.Context, _, user string) (string, error) {
	fn := user[strings.LastIndex(user, "Função:\n```go\n")+len("Função:\n```go\n"):]
	fn = strings.TrimSuffix(fn, "\n```\n")
	end := strings.LastIndex(fn, "}")
	body := strings.Index(fn[strings.Index(fn, "func"):], "{\n") + strings.Index(fn, "func") + 2
	out, _ := json.Marshal(map[string]string{
		"pedido":          "Escreva a função.",
		"alteracao":       "Acrescente uma instrução.",
		"codigo_alterado": fn[:body] + "\t_ = 1\n" + fn[body:],
		"codigo_quebrado": fn[:end] + "\tnaoExiste()\n}",
	})
	return string(out), nil
}

// TestTeachOnCorpus roda a geração inteira sobre o gorilla/mux do corpus, com
// o compilador de verdade. Sem o corpus baixado, o teste é pulado.
func TestTeachOnCorpus(t *testing.T) {
	root, _ := filepath.Abs("../../corpus")
	src := filepath.Join(root, "src", "gorilla__mux")
	db := filepath.Join(root, "db", "gorilla__mux.db")
	if _, err := os.Stat(db); err != nil || testing.Short() {
		t.Skip("corpus ausente")
	}
	st, err := store.Open(db)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	var out bytes.Buffer
	rep, err := Teach(context.Background(), st, TeachOptions{
		Root: src, Source: "gorilla/mux", Backend: fakeTeacher{}, Workers: 4,
		N: 12, MinLines: 5, MaxLines: 40, Env: (&corpus.Corpus{Dir: root}).GoEnv(), Out: &out,
	})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Create != 12 || rep.Edit < 10 || rep.Fix < 10 {
		t.Errorf("exemplos: %+v", rep)
	}
	for _, line := range strings.Split(strings.TrimSpace(out.String()), "\n") {
		var r Record
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatal(err)
		}
		if r.Kind == "fix" && !strings.Contains(r.Input, "undefined: naoExiste") {
			t.Errorf("%s: erro do compilador ausente:\n%s", r.Name, r.Input)
		}
		if r.Kind == "fix" && strings.Contains(r.Input, "- fora da função") {
			t.Errorf("%s: linha do erro fora da função:\n%s", r.Name, r.Input)
		}
	}
	if testing.Verbose() {
		t.Log(out.String())
	}
}
