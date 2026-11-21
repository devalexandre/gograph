package implement

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/devalexandre/gograph/internal/describe"
	"github.com/devalexandre/gograph/internal/store"
)

// TeacherPrompt orienta o modelo professor a produzir, para uma função real, o
// material de três tarefas: escrever a partir de um pedido, alterar e corrigir.
const TeacherPrompt = `Você prepara exemplos de treino para um modelo pequeno que programa em Go. Você recebe o contexto de um arquivo (imports, tipos, funções disponíveis no projeto) e uma função real. Responda com um objeto JSON com quatro campos:

- "pedido": o pedido que um desenvolvedor faria a um colega para que ele escrevesse esta função do zero. De 2 a 4 frases em português, diretas. Diga o que a função recebe e devolve, o comportamento, os casos especiais e de erro que o código trata. Não cite nomes de variáveis locais nem descreva o código linha a linha.
- "alteracao": um pedido curto e realista, em português, de uma mudança pequena de comportamento nesta função (tratar um caso a mais, validar uma entrada, mudar uma mensagem de erro, trocar um valor padrão, registrar um log, sair mais cedo). A mudança não pode alterar a assinatura.
- "codigo_alterado": a função completa depois da alteração pedida, com o doc comment atualizado se ele existir. Mude só o necessário. Use apenas pacotes já importados no arquivo, a biblioteca padrão e os tipos e funções mostrados no contexto.
- "codigo_quebrado": a função original com 1 a 3 erros de compilação do tipo que um modelo pequeno comete: campo ou método que não existe no tipo, função inexistente ou do pacote errado, número ou tipo de argumentos errado, valor de retorno faltando ou sobrando, variável declarada e não usada, tipo incompatível numa atribuição. Não escreva comentário apontando o erro e não mude o restante da função.

Nos dois campos de código escreva só a função, sem cerca de código.`

func teacherSchema() map[string]any {
	str := map[string]any{"type": "string"}
	fields := []string{"pedido", "alteracao", "codigo_alterado", "codigo_quebrado"}
	props := map[string]any{}
	for _, f := range fields {
		props[f] = str
	}
	return map[string]any{"type": "object", "properties": props, "required": fields, "additionalProperties": false}
}

// NewTeacher devolve o backend da Anthropic configurado para TeacherPrompt.
func NewTeacher(model, effort string) *describe.Anthropic {
	a := describe.NewAnthropic(model, effort)
	a.Format = teacherSchema()
	return a
}

// TeachOptions configura a geração num repositório.
type TeachOptions struct {
	Root, Source, License string
	Backend               describe.Backend
	Workers               int
	N, MinLines, MaxLines int
	Go                    string
	Env                   []string
	Done                  map[string]bool // hashes já gerados numa rodada anterior
	Out                   io.Writer       // JSONL com os exemplos
	Progress              func(name string, kinds []string, err error)
}

// TeachReport conta os exemplos gravados por tarefa.
type TeachReport struct {
	Functions, Failed int
	Create, Edit, Fix int
}

// Teach pede ao professor o material das funções sorteadas e grava os
// exemplos. O compilador confere o que o professor escreveu: a alteração só
// entra se compilar, e a versão quebrada só entra se de fato não compilar,
// com os erros reais do compilador.
func Teach(ctx context.Context, st *store.Store, opts TeachOptions) (*TeachReport, error) {
	funcs, err := candidates(st, opts.Root, opts.N, opts.MinLines, opts.MaxLines, false)
	if err != nil {
		return nil, err
	}
	if opts.Workers < 1 {
		opts.Workers = 1
	}
	bopts := Options{Root: opts.Root, Go: opts.Go, Env: opts.Env}
	rep := &TeachReport{}
	var mu sync.Mutex // protege rep, Out e builds
	builds := map[string]bool{}
	// Pacote que não compila nem no original não serve para conferir nada.
	pkgBuilds := func(pkg string) bool {
		mu.Lock()
		defer mu.Unlock()
		ok, seen := builds[pkg]
		if !seen {
			_, err := goCmd(ctx, bopts, 5*time.Minute, "build", "-o", os.DevNull, "./"+pkg)
			ok = err == nil
			builds[pkg] = ok
		}
		return ok
	}

	jobs := make(chan store.FuncRow)
	var wg sync.WaitGroup
	for w := 0; w < opts.Workers; w++ {
		work, err := os.MkdirTemp("", "gograph-teach-")
		if err != nil {
			return nil, err
		}
		defer os.RemoveAll(work)
		wg.Add(1)
		go func() {
			defer wg.Done()
			for f := range jobs {
				recs, err := teachOne(ctx, st, opts, bopts, work, f)
				var kinds []string
				mu.Lock()
				rep.Functions++
				if err != nil {
					rep.Failed++
				}
				for _, r := range recs {
					line, _ := json.Marshal(r)
					fmt.Fprintf(opts.Out, "%s\n", line)
					kinds = append(kinds, r.Kind)
					switch r.Kind {
					case "create":
						rep.Create++
					case "edit":
						rep.Edit++
					case "fix":
						rep.Fix++
					}
				}
				if opts.Progress != nil {
					opts.Progress(f.Name, kinds, err)
				}
				mu.Unlock()
			}
		}()
	}
	for _, f := range funcs {
		if ctx.Err() != nil {
			break
		}
		if opts.Done[f.Hash] || !pkgBuilds(filepath.Dir(f.File)) {
			continue
		}
		jobs <- f
	}
	close(jobs)
	wg.Wait()
	return rep, ctx.Err()
}

func teachOne(ctx context.Context, st *store.Store, opts TeachOptions, bopts Options, work string, f store.FuncRow) ([]Record, error) {
	task, err := BuildTask(opts.Root, st.DB(), f)
	if err != nil {
		return nil, err
	}
	user := task.Context + "\nFunção:\n```go\n" + task.Doc + task.Original + "\n```\n"
	raw, err := opts.Backend.Generate(ctx, TeacherPrompt, user)
	if err != nil {
		return nil, err
	}
	var t struct {
		Pedido   string `json:"pedido"`
		Mudanca  string `json:"alteracao"`
		Alterado string `json:"codigo_alterado"`
		Quebrado string `json:"codigo_quebrado"`
	}
	if err := json.Unmarshal([]byte(raw), &t); err != nil {
		return nil, fmt.Errorf("resposta do professor inválida: %w", err)
	}
	if strings.TrimSpace(t.Pedido) == "" {
		return nil, fmt.Errorf("professor não escreveu o pedido")
	}
	rec := func(kind, system, input, output string) Record {
		return Record{Source: opts.Source, License: opts.License, Name: f.Name, Hash: f.Hash,
			Kind: kind, System: system, Input: input, Output: output}
	}
	recs := []Record{rec("create", SystemPrompt, task.RequestPrompt(t.Pedido), task.Original)}

	// Alteração: vale se compilar e se de fato mudar a função.
	if code := ExtractCode(t.Alterado); strings.HasPrefix(code, "func") && strings.TrimSpace(t.Mudanca) != "" {
		if c, err := compile(ctx, bopts, work, task, code); err == nil && c.fn != "" && !sameCode(c.fn, task.Original) {
			recs = append(recs, rec("edit", EditSystemPrompt, task.EditPrompt(t.Mudanca), docOf(t.Alterado)+c.fn))
		}
	}
	// Correção: vale se não compilar e o compilador apontar o erro.
	if code := ExtractCode(t.Quebrado); strings.HasPrefix(code, "func") {
		// Erro que fala da versão da linguagem no go.mod não ensina nada sobre a função.
		if c, err := compile(ctx, bopts, work, task, code); err != nil && len(c.errs) > 0 && c.fn != "" &&
			!strings.Contains(strings.Join(c.errs, "\n"), "-lang was set") {
			recs = append(recs, rec("fix", FixSystemPrompt, FixPrompt(task.Prompt, c.fn, c.errs), task.Original))
		}
	}
	return recs, ctx.Err()
}

// docOf devolve as linhas de comentário que antecedem o func na resposta.
func docOf(answer string) string {
	var doc string
	for _, l := range strings.Split(strings.TrimSpace(answer), "\n") {
		if !strings.HasPrefix(strings.TrimSpace(l), "//") {
			break
		}
		doc += l + "\n"
	}
	return doc
}

// sameCode compara ignorando espaços, para não aceitar como alteração uma
// função que só mudou de formatação.
func sameCode(a, b string) bool {
	return strings.Join(strings.Fields(a), "") == strings.Join(strings.Fields(b), "")
}

// contextEnd diz, para cada tipo de exemplo, onde o contexto termina na entrada.
var contextEnd = map[string]string{
	"":       "\nEscreva a função:\n",
	"create": "\nPedido: ",
	"edit":   "\nFunção atual:\n",
	"fix":    "\nEscreva a função:\n",
}

// Recontext refaz o contexto (tipos, constantes, funções disponíveis) da
// entrada de um exemplo já gerado, mantendo o pedido e a resposta. Serve para
// aproveitar os exemplos do professor quando o formato do contexto muda.
func Recontext(st *store.Store, root string, r *Record) error {
	var f store.FuncRow
	err := st.DB().QueryRow(`
		SELECT id, name, package, file, line_start, line_end, signature, hash
		FROM functions WHERE hash = ? AND name = ? AND external = 0`, r.Hash, r.Name).
		Scan(&f.ID, &f.Name, &f.Package, &f.File, &f.LineStart, &f.LineEnd, &f.Signature, &f.Hash)
	if err != nil {
		return fmt.Errorf("%s: função não está mais no grafo: %w", r.Name, err)
	}
	task, err := BuildTask(root, st.DB(), f)
	if err != nil {
		return err
	}
	i := strings.Index(r.Input, contextEnd[r.Kind])
	if i < 0 {
		return fmt.Errorf("%s: entrada sem a marca do fim do contexto", r.Name)
	}
	r.Input = task.Context + r.Input[i:]
	return nil
}
