package describe

import (
	"os"
	"regexp"
	"testing"
)

func TestParse(t *testing.T) {
	allowed := map[string]bool{"repo.Save": true}
	valid := `{"purpose":"Valida o pedido e encaminha para persistência.","calls":[{"fn":"repo.Save","why":"persistir o pedido"}],"effects":[],"errors":[],"concurrency":""}`
	if _, err := Parse("```json\n"+valid+"\n```", allowed); err != nil {
		t.Errorf("descrição válida rejeitada: %v", err)
	}
	bad := map[string]string{
		"não é JSON":           `purpose: x`,
		"purpose vazio":        `{"purpose":" ","calls":[],"effects":[],"errors":[],"concurrency":""}`,
		"purpose longo":        `{"purpose":"a a a a a a a a a a a a a a a a a a a a a a a a a a","calls":[],"effects":[],"errors":[],"concurrency":""}`,
		"callee fora da lista": `{"purpose":"x","calls":[{"fn":"repo.Delete","why":"y"}],"effects":[],"errors":[],"concurrency":""}`,
		"campo desconhecido":   `{"purpose":"x","calls":[],"effects":[],"errors":[],"concurrency":"","extra":1}`,
		"chamada sem intenção": `{"purpose":"x","calls":[{"fn":"repo.Save","why":""}],"effects":[],"errors":[],"concurrency":""}`,
	}
	for name, raw := range bad {
		if _, err := Parse(raw, allowed); err == nil {
			t.Errorf("%s: deveria ser rejeitado", name)
		}
	}
}

func TestShortName(t *testing.T) {
	if got := ShortName("github.com/a/b/repo.(*T).Save", "github.com/a/b/repo"); got != "repo.(*T).Save" {
		t.Errorf("ShortName = %q", got)
	}
	if got := ShortName("error.Error", ""); got != "error.Error" {
		t.Errorf("ShortName = %q", got)
	}
}

// O prompt dos modelos treinados precisa ser idêntico ao usado no treino.
func TestTunedSystemPromptMatchesTraining(t *testing.T) {
	src, err := os.ReadFile("../../train/prepare.py")
	if err != nil {
		t.Skip(err)
	}
	m := regexp.MustCompile(`(?s)SYSTEM = \((.*?)\n\)`).FindSubmatch(src)
	if m == nil {
		t.Fatal("SYSTEM não encontrado em prepare.py")
	}
	var got string
	for _, part := range regexp.MustCompile(`"([^"]*)"`).FindAllSubmatch(m[1], -1) {
		got += string(part[1])
	}
	if got != TunedSystemPrompt {
		t.Errorf("prompt de treino diferente do de inferência:\n%q\n%q", got, TunedSystemPrompt)
	}
}

func TestRepair(t *testing.T) {
	allowed := map[string]bool{"repo.Save": true}
	raw := `{"purpose":"Valida e persiste o pedido.","calls":[{"fn":"regexp.CompileFunc","why":"inventada"},{"fn":"repo.Save","why":"persistir"},{"fn":"x.Y","why":"outra inventada"}],"effects":[],"errors":["Não devolve erro; sempre retorna true","repassa erro de Save"],"concurrency":""}`
	if _, err := Parse(raw, allowed); err == nil {
		t.Fatal("Parse deveria rejeitar chamada fora da lista")
	}
	d, err := Repair(raw, allowed)
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Calls) != 1 || d.Calls[0].Fn != "repo.Save" {
		t.Errorf("calls não corrigido: %+v", d.Calls)
	}
	if len(d.Errors) != 1 || d.Errors[0] != "repassa erro de Save" {
		t.Errorf("negação não removida de errors: %q", d.Errors)
	}
	if _, err := Repair(`{"purpose":""}`, allowed); err == nil {
		t.Error("Repair não deve aceitar descrição sem purpose")
	}
}
