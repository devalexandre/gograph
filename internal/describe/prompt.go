package describe

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/devalexandre/gograph/internal/store"
)

// SystemPrompt são as regras do formato da descrição. Mudar este texto ou o
// esquema invalida o dataset e os modelos treinados com ele.
const SystemPrompt = `Você descreve funções Go para um índice de código. Cada descrição é guardada junto de um grafo de chamadas: quem lê a sua descrição já consegue navegar para as funções chamadas e ler a descrição delas. Por isso a descrição fala só do que esta função faz localmente.

Responda com um único objeto JSON com estes campos:

- "purpose": uma frase, com no máximo 20 palavras, dizendo para que a função serve. Diga o que ela resolve, não repita o nome dela em outras palavras: para processOrder, "processa o pedido" não informa nada; "valida o pedido e encaminha para persistência" informa.
- "calls": as chamadas mais relevantes para entender a função, no máximo 5, cada uma com "fn" (o nome exatamente como aparece na lista de chamadas recebida) e "why" (a intenção da chamada do ponto de vista desta função, por exemplo "persistir o pedido"). Descreva por que a função chama, não como a função chamada funciona por dentro: se ela mudar, a sua descrição precisa continuar correta. Deixe de fora chamadas de log, formatação e conversão que não ajudam a entender o fluxo. Use lista vazia se não houver chamada relevante.
- "effects": efeitos colaterais que a própria função causa diretamente no seu corpo, como escrever em arquivo, executar uma query, alterar estado global ou iniciar uma goroutine. Efeitos que acontecem dentro das funções chamadas não entram. Use lista vazia se não houver.
- "errors": os erros que a função devolve e quando, por exemplo "ErrInvalidOrder quando a validação falha" ou "repassa erro de Save". Use lista vazia se a função não devolve erro.
- "concurrency": uma frase sobre concorrência quando houver algo a dizer (goroutines, canais, mutex, uso de context para cancelamento). Use string vazia caso contrário.

Escreva em português, de forma direta. Baseie-se apenas no código recebido; não invente comportamento que não está nele.`

// TunedSystemPrompt é o prompt dos modelos treinados (train/prepare.py, SYSTEM).
// Eles aprenderam as regras pelos exemplos e foram treinados com este texto
// exato; trocar por outro piora a saída.
const TunedSystemPrompt = "Você descreve funções Go para um índice de código. Responda com um objeto " +
	"JSON com os campos purpose, calls, effects, errors e concurrency, em " +
	"português, falando só do que a função faz localmente."

// maxCalleesInPrompt limita a lista de chamadas enviada ao modelo.
const maxCalleesInPrompt = 40

// ShortName troca o caminho de import completo pelo nome do pacote:
// "github.com/a/b/repo.(*T).Save" vira "repo.(*T).Save".
func ShortName(name, pkg string) string {
	if pkg == "" || !strings.HasPrefix(name, pkg) {
		return name
	}
	return path.Base(pkg) + name[len(pkg):]
}

// Input é o texto enviado ao modelo para uma função.
type Input struct {
	Text    string
	Allowed map[string]bool // nomes que podem aparecer em calls
	Lines   int
}

// BuildInput monta a entrada do modelo: assinatura, doc comment, corpo e a
// lista de chamadas com o purpose de cada uma.
func BuildInput(root string, f store.FuncRow, callees []store.Callee) (*Input, error) {
	doc, body, err := readSource(filepath.Join(root, f.File), f.LineStart, f.LineEnd)
	if err != nil {
		return nil, err
	}
	in := &Input{Allowed: map[string]bool{}, Lines: f.LineEnd - f.LineStart + 1}

	var b strings.Builder
	fmt.Fprintf(&b, "Função: %s\n", ShortName(f.Name, f.Package))
	fmt.Fprintf(&b, "Assinatura: %s\n", f.Signature)
	if doc != "" {
		fmt.Fprintf(&b, "\nDoc comment:\n%s\n", doc)
	}
	fmt.Fprintf(&b, "\nCódigo:\n```go\n%s\n```\n", body)

	seen := map[string]bool{}
	var lines []string
	for _, c := range callees {
		if c.ID == f.ID {
			continue
		}
		short := ShortName(c.Name, c.Package)
		if seen[short] {
			continue
		}
		seen[short] = true
		in.Allowed[short] = true
		if len(lines) >= maxCalleesInPrompt {
			continue
		}
		line := fmt.Sprintf("- %s [%s]", short, c.Kind)
		if p := Purpose(c.Description); p != "" {
			line += ": " + p
		} else if c.External && c.Signature != "" {
			line += " " + c.Signature
		}
		lines = append(lines, line)
	}
	if len(lines) == 0 {
		b.WriteString("\nChamadas: nenhuma.\n")
	} else {
		fmt.Fprintf(&b, "\nChamadas (nome [tipo]: para que serve):\n%s\n", strings.Join(lines, "\n"))
		if extra := len(seen) - len(lines); extra > 0 {
			fmt.Fprintf(&b, "(e mais %d chamadas não listadas)\n", extra)
		}
	}
	in.Text = b.String()
	return in, nil
}

// readSource devolve o doc comment logo acima da função e as linhas do corpo.
func readSource(file string, start, end int) (doc, body string, err error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return "", "", err
	}
	lines := strings.Split(string(data), "\n")
	if start < 1 || end > len(lines) || start > end {
		return "", "", fmt.Errorf("%s: linhas %d-%d fora do arquivo; rode o index de novo", file, start, end)
	}
	body = strings.Join(lines[start-1:end], "\n")

	first := start - 1
	for first > 0 && strings.HasPrefix(strings.TrimSpace(lines[first-1]), "//") {
		first--
	}
	doc = strings.Join(lines[first:start-1], "\n")
	return doc, body, nil
}
