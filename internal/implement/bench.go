package implement

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/devalexandre/gograph/internal/describe"
	"github.com/devalexandre/gograph/internal/store"
)

// OriginalModel, no lugar do modelo, devolve o código original da função.
// Serve para conferir o medidor: o resultado tem de ser 100%.
const OriginalModel = "original"

// TestSample é o tamanho da amostra de teste por repositório. A amostra de
// validação (Options.Tail) exclui essas funções.
const TestSample = 60

// Options configura uma rodada do medidor.
type Options struct {
	Root     string // raiz do módulo
	Host     string // endereço do Ollama
	Model    string // modelo no Ollama
	N        int    // funções a testar
	MinLines int    // tamanho mínimo da função
	MaxLines int    // tamanho máximo da função
	Tests    bool   // roda go test nas funções cobertas por teste
	Retries  int    // vezes que o modelo recebe o erro do compilador para corrigir
	Tail     bool   // sorteia do fim da ordem: funções que a amostra normal não pega
	// Dump só grava as tarefas (entrada de cada função), sem chamar modelo nem
	// compilar: é o que um gerador externo precisa para produzir as respostas.
	Dump bool
	// Answers são respostas já geradas, pelo hash da função. Quando não é nil
	// o Ollama não é chamado: serve para medir um modelo que roda fora dele,
	// como um checkpoint no meio do treino.
	Answers       map[string]string
	AnswerTimings map[string]int64
	GoImports     bool            // aplica goimports ao arquivo antes do go build
	Vet           bool            // roda go vet após go build
	Workers       int             // número de workers em paralelo (padrão 1)
	SkipHashes    map[string]bool // hashes já concluídos (para retomada)
	Go            string          // binário do Go; vazio = "go" do PATH
	Env           []string  // ambiente do go build/test; nil = o do processo
	Out           io.Writer // se não for nil, recebe um JSONL com cada tentativa
	Progress      func(r Result)
}

// FixAttempt registra uma tentativa de correção automática.
type FixAttempt struct {
	Attempt  int    `json:"attempt"`
	Trigger  string `json:"trigger"`  // "build", "vet" ou "test"
	Category string `json:"category"` // categoria do erro que motivou a tentativa
}

// Result é o desfecho de uma função.
type Result struct {
	Name   string `json:"name"`
	Hash   string `json:"hash"`
	Prompt string `json:"prompt"`
	System string `json:"system,omitempty"` // só no modo Dump
	Answer string `json:"answer"`
	Built  bool   `json:"built"`
	Fixed  bool   `json:"fixed,omitempty"` // só compilou/passou depois de receber o erro
	// Code e Errors descrevem a tentativa que não compilou: a função como o
	// compilador a viu e os erros com a linha contada dentro dela.
	Code     string   `json:"code,omitempty"`
	Errors   []string `json:"errors,omitempty"`
	Original string   `json:"original,omitempty"`
	// FixSystem e FixPrompt são o pedido de correção desta tentativa, prontos
	// para um gerador externo fazer a segunda tentativa.
	FixSystem   string       `json:"fix_system,omitempty"`
	FixPrompt   string       `json:"fix_prompt,omitempty"`
	PassAttempt string       `json:"pass_attempt,omitempty"` // "0", "1", "2" ou "nunca"
	FixAttempts []FixAttempt `json:"fix_attempts,omitempty"`
	Covered     bool         `json:"covered"` // exercitada pelos testes do pacote
	Passed      bool         `json:"passed"`  // testes do pacote passaram com o código gerado
	Truncated   bool         `json:"truncated,omitempty"`
	InfraError  bool         `json:"infra_error,omitempty"`
	ElapsedMs   int64        `json:"elapsed_ms,omitempty"`
	Error       string       `json:"error,omitempty"`
}

// Report resume a rodada.
type Report struct {
	Total, Built    int
	Fixed           int // das que compilam, quantas precisaram de correção
	Skipped         int // funções em pacotes que não compilam nem no original
	Covered, Passed int
	Elapsed         time.Duration
}

type covRange struct {
	start, end int
	testName   string
}

// pkgTests guarda o que os testes de um pacote cobrem no código original.
type pkgTests struct {
	ok      bool
	covered map[string][]covRange // arquivo (base) -> intervalos de linhas exercitados por cada teste
}

// Run pede ao modelo cada função da amostra e confere o resultado com
// go build e, quando a função é coberta por teste, com go test. O código
// gerado entra por -overlay: nenhum arquivo do projeto é alterado.
func Run(ctx context.Context, st *store.Store, opts Options) (*Report, error) {
	start := time.Now()
	funcs, err := candidates(st, opts.Root, opts.N, opts.MinLines, opts.MaxLines, opts.Tail)
	if err != nil {
		return nil, err
	}
	work, err := os.MkdirTemp("", "gograph-bench-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(work)

	rep := &Report{}
	tests := map[string]*pkgTests{}
	builds := map[string]bool{}
	type workItem struct {
		f         store.FuncRow
		task      *Task
		testNames []string
		covered   bool
	}
	var items []workItem
	for _, f := range funcs {
		if ctx.Err() != nil {
			break
		}
		if opts.SkipHashes != nil && opts.SkipHashes[f.Hash] {
			continue
		}
		// Pacote que já não compila no original (versão do Go, dependência
		// ausente) não mede o modelo; fica fora da conta.
		pkg := filepath.Dir(f.File)
		ok, seen := builds[pkg]
		if !seen {
			_, berr := goCmd(ctx, opts, 5*time.Minute, "build", "-o", os.DevNull, "./"+pkg)
			ok = berr == nil
			builds[pkg] = ok
		}
		if !ok {
			rep.Skipped++
			continue
		}
		task, err := BuildTask(opts.Root, st.DB(), f)
		if err != nil {
			continue
		}
		if opts.Dump {
			r := Result{Name: f.Name, Hash: f.Hash, Prompt: task.Prompt, System: SystemPrompt}
			rep.Total++
			if opts.Out != nil {
				line, _ := json.Marshal(r)
				fmt.Fprintf(opts.Out, "%s\n", line)
			}
			continue
		}
		dir := filepath.Dir(f.File)
		var testNames []string
		var covered bool
		if opts.Tests {
			pt, ok := tests[dir]
			if !ok {
				pt = baseline(ctx, opts, work, dir)
				tests[dir] = pt
			}
			if pt.ok {
				testNames = pt.testsFor(filepath.Base(f.File), f.LineStart+1, f.LineEnd)
				covered = len(testNames) > 0
			}
		}
		items = append(items, workItem{f: f, task: task, testNames: testNames, covered: covered})
	}

	workers := opts.Workers
	if workers < 1 {
		workers = 1
	}
	if workers > len(items) && len(items) > 0 {
		workers = len(items)
	}
	ch := make(chan workItem, len(items))
	for _, it := range items {
		ch <- it
	}
	close(ch)

	var mu sync.Mutex
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			wdir, err := os.MkdirTemp(work, "w-")
			if err != nil {
				wdir = work
			}
			for it := range ch {
				if ctx.Err() != nil {
					break
				}
				f, task, testNames := it.f, it.task, it.testNames
				r := Result{Name: f.Name, Hash: f.Hash, Prompt: task.Prompt, Original: task.Original, Covered: it.covered}

				fnStart := time.Now()
				var inferMs int64
				var truncated bool
				var err error
				if opts.Answers != nil {
					answer, ok := opts.Answers[f.Hash]
					if !ok {
						err = fmt.Errorf("sem resposta para esta função no arquivo de respostas")
					}
					r.Answer = answer
					if opts.AnswerTimings != nil {
						inferMs = opts.AnswerTimings[f.Hash]
					}
				} else if opts.Model == OriginalModel {
					r.Answer, err = task.Original, nil
				} else {
					r.Answer, truncated, err = chat(ctx, opts, SystemPrompt, task.Prompt)
					inferMs = time.Since(fnStart).Milliseconds()
					if truncated {
						r.Truncated = true
					}
				}
				var infra *infraErr
				if err != nil && errors.As(err, &infra) {
					r.InfraError = true
				}
				var trigger, testOut string
				if err == nil {
					trigger, testOut, err = checkAttempt(ctx, opts, wdir, task, &r, testNames)
				}
				passTry := 0
				for try := 1; !r.InfraError && err != nil && (truncated || len(r.Errors) > 0 || testOut != "") && try <= opts.Retries; try++ {
					firstErr := ""
					if len(r.Errors) > 0 {
						firstErr = r.Errors[0]
					}
					curTrigger := trigger
					if truncated {
						curTrigger = "length"
					}
					r.FixAttempts = append(r.FixAttempts, FixAttempt{
						Attempt:  try,
						Trigger:  curTrigger,
						Category: ClassifyError(curTrigger, firstErr),
					})
					tStart := time.Now()
					var answer string
					var cerr error
					if truncated {
						answer, truncated, cerr = chat(ctx, opts, SystemPrompt, task.Prompt)
					} else {
						prompt := ""
						if trigger == "test" {
							prompt = TestFixPrompt(task.Prompt, r.Code, testOut)
						} else {
							prompt = FixPrompt(task.Prompt, r.Code, r.Errors)
						}
						answer, truncated, cerr = chat(ctx, opts, FixSystemPrompt, prompt)
					}
					inferMs += time.Since(tStart).Milliseconds()
					if cerr != nil {
						if errors.As(cerr, &infra) {
							r.InfraError = true
							err = cerr
						}
						break
					}
					r.Truncated = truncated
					r.Answer, r.Code, r.Errors, r.Built, r.Passed = answer, "", nil, false, false
					trigger, testOut, err = checkAttempt(ctx, opts, wdir, task, &r, testNames)
					if err == nil {
						r.Fixed = true
						passTry = try
					}
				}
				if opts.Retries > 0 && !r.InfraError {
					if err == nil {
						r.PassAttempt = strconv.Itoa(passTry)
					} else {
						r.PassAttempt = "nunca"
					}
				}
				r.ElapsedMs = inferMs
				if err != nil {
					r.Error = err.Error()
				}
				if !r.Built && len(r.Errors) > 0 && !r.InfraError {
					r.FixSystem, r.FixPrompt = FixSystemPrompt, FixPrompt(task.Prompt, r.Code, r.Errors)
				}

				mu.Lock()
				rep.Total++
				if r.Built {
					rep.Built++
				}
				if r.Fixed {
					rep.Fixed++
				}
				if r.Covered {
					rep.Covered++
					if r.Passed {
						rep.Passed++
					}
				}
				if opts.Out != nil {
					line, _ := json.Marshal(r)
					fmt.Fprintf(opts.Out, "%s\n", line)
					if outFile, ok := opts.Out.(*os.File); ok {
						_ = outFile.Sync()
					}
				}
				if opts.Progress != nil {
					opts.Progress(r)
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	rep.Elapsed = time.Since(start)
	return rep, ctx.Err()
}

// candidates sorteia as funções: a ordem pelo hash dá sempre a mesma amostra.
// Ficam de fora testes, vendor, mocks, código gerado, closures, main e init.
func candidates(st *store.Store, root string, n, minLines, maxLines int, tail bool) ([]store.FuncRow, error) {
	order := "hash"
	// A ponta de trás serve de validação: nunca pode repetir uma função da
	// amostra normal, que é a do teste. Num repositório pequeno as duas
	// pontas se encontram, então as primeiras TestSample ficam de fora.
	head := map[int64]bool{}
	if tail {
		order = "hash DESC"
		first, err := candidates(st, root, TestSample, minLines, maxLines, false)
		if err != nil {
			return nil, err
		}
		for _, f := range first {
			head[f.ID] = true
		}
	}
	rows, err := st.DB().Query(`
		SELECT id, name, package, file, line_start, line_end, signature, hash
		FROM functions
		WHERE external = 0 AND hash <> '' AND signature <> '' AND name NOT LIKE '%$%'
		  AND file NOT LIKE '%_test.go' AND file NOT LIKE '%vendor/%' AND file NOT LIKE '%mock%'
		  AND line_end - line_start + 1 BETWEEN ? AND ?
		ORDER BY `+order, minLines, maxLines)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	generated := map[string]bool{}
	var out []store.FuncRow
	for rows.Next() && len(out) < n {
		var f store.FuncRow
		if err := rows.Scan(&f.ID, &f.Name, &f.Package, &f.File, &f.LineStart, &f.LineEnd, &f.Signature, &f.Hash); err != nil {
			return nil, err
		}
		if strings.HasSuffix(f.Name, ".main") || strings.HasSuffix(f.Name, ".init") || head[f.ID] {
			continue
		}
		g, ok := generated[f.File]
		if !ok {
			g = describe.GeneratedFile(filepath.Join(root, f.File))
			generated[f.File] = g
		}
		if !g {
			out = append(out, f)
		}
	}
	return out, rows.Err()
}

// Record é um exemplo de treino da tarefa de escrita.
type Record struct {
	Source  string `json:"source,omitempty"`
	License string `json:"license,omitempty"`
	Name    string `json:"name"`
	Hash    string `json:"hash"`
	Kind    string `json:"kind,omitempty"` // create, edit ou fix; vazio = escrita pela assinatura
	System  string `json:"system"`
	Input   string `json:"input"`
	Output  string `json:"output"`
}

// Export grava em w até n exemplos de treino tirados do projeto em root: a
// entrada é a de BuildTask e a saída é a função original.
func Export(st *store.Store, root, source, license string, n, minLines, maxLines int, w io.Writer) (int, error) {
	funcs, err := candidates(st, root, n, minLines, maxLines, false)
	if err != nil {
		return 0, err
	}
	written := 0
	for _, f := range funcs {
		task, err := BuildTask(root, st.DB(), f)
		if err != nil {
			continue
		}
		line, err := json.Marshal(Record{
			Source: source, License: license, Name: f.Name, Hash: f.Hash,
			System: SystemPrompt, Input: task.Prompt, Output: task.Original,
		})
		if err != nil {
			return written, err
		}
		if _, err := fmt.Fprintf(w, "%s\n", line); err != nil {
			return written, err
		}
		written++
	}
	return written, nil
}

func goCmd(ctx context.Context, opts Options, timeout time.Duration, args ...string) (string, error) {
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	gobin := opts.Go
	if gobin == "" {
		gobin = "go"
	}
	cmd := exec.CommandContext(cctx, gobin, args...)
	cmd.Dir = opts.Root
	cmd.Env = opts.Env
	// O go test dispara outro processo (o binário de teste). Matar só o go
	// deixaria um teste travado rodando; o grupo inteiro precisa cair junto.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
	cmd.WaitDelay = 5 * time.Second
	out, err := cmd.CombinedOutput()
	if cctx.Err() == context.DeadlineExceeded {
		return "", fmt.Errorf("tempo esgotado em go %s", args[0])
	}
	return string(out), err
}

// baseline roda os testes do pacote sobre o código original e registra as
// linhas exercitadas. Pacote sem teste, com teste falhando ou lento demais
// fica só com a verificação de compilação.
func baseline(ctx context.Context, opts Options, work, dir string) *pkgTests {
	pt := &pkgTests{covered: map[string][]covRange{}}
	listOut, err := goCmd(ctx, opts, 2*time.Minute, "test", "-list", "^Test", "./"+dir)
	if err != nil {
		return pt
	}
	var testNames []string
	for _, line := range strings.Split(listOut, "\n") {
		name := strings.TrimSpace(line)
		if strings.HasPrefix(name, "Test") {
			testNames = append(testNames, name)
		}
	}
	if len(testNames) == 0 {
		return pt
	}
	testBin := filepath.Join(work, "pkg.test")
	os.Remove(testBin)
	if _, err := goCmd(ctx, opts, 4*time.Minute, "test", "-c", "-cover", "-o", testBin, "./"+dir); err != nil {
		return pt
	}
	defer os.Remove(testBin)

	profile := filepath.Join(work, "cover.out")
	pkgDir := filepath.Join(opts.Root, dir)
	for _, tname := range testNames {
		if ctx.Err() != nil {
			break
		}
		os.Remove(profile)
		tctx, cancel := context.WithTimeout(ctx, 180*time.Second)
		cmd := exec.CommandContext(tctx, testBin, "-test.run=^"+tname+"$", "-test.timeout=180s", "-test.coverprofile="+profile)
		cmd.Dir = pkgDir
		cmd.Env = opts.Env
		cmd.WaitDelay = 5 * time.Second
		runErr := cmd.Run()
		cancel()
		if runErr != nil {
			continue
		}
		f, err := os.Open(profile)
		if err != nil {
			continue
		}
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			// formato: caminho/arquivo.go:10.2,12.16 3 1
			line := sc.Text()
			colon := strings.LastIndexByte(line, ':')
			fields := strings.Fields(line[colon+1:])
			if colon < 0 || len(fields) != 3 || fields[2] == "0" {
				continue
			}
			from, to, ok := strings.Cut(fields[0], ",")
			if !ok {
				continue
			}
			a, _ := strconv.Atoi(strings.Split(from, ".")[0])
			b, _ := strconv.Atoi(strings.Split(to, ".")[0])
			base := filepath.Base(line[:colon])
			pt.covered[base] = append(pt.covered[base], covRange{start: a, end: b, testName: tname})
		}
		f.Close()
	}
	pt.ok = len(pt.covered) > 0
	return pt
}

func (pt *pkgTests) testsFor(file string, start, end int) []string {
	seen := map[string]bool{}
	var out []string
	for _, r := range pt.covered[file] {
		if r.start <= end && r.end >= start && !seen[r.testName] {
			seen[r.testName] = true
			out = append(out, r.testName)
		}
	}
	sort.Strings(out)
	return out
}

// checkAttempt compila, roda vet (se habilitado) e testa o pacote com a função gerada.
func checkAttempt(ctx context.Context, opts Options, work string, task *Task, r *Result, testNames []string) (string, string, error) {
	code := ExtractCode(r.Answer)
	if !strings.HasPrefix(code, "func") {
		r.Code = code
		r.Errors = []string{"resposta sem função"}
		return "build", "", fmt.Errorf("resposta sem função")
	}
	b, trigger, err := compileWithTrigger(ctx, opts, work, task, code)
	if err != nil {
		r.Code, r.Errors = b.fn, b.errs
		if len(r.Errors) == 0 {
			r.Errors = []string{err.Error()}
		}
		return trigger, "", err
	}
	overlayFile := b.overlay
	dir := "./" + filepath.Dir(task.Func.File)
	r.Built = true
	if !r.Covered {
		return "", "", nil
	}
	args := []string{"test", "-count=1", "-timeout=180s", "-overlay=" + overlayFile}
	if len(testNames) > 0 && len(testNames) <= 200 {
		args = append(args, "-run=^("+strings.Join(testNames, "|")+")$")
	}
	args = append(args, dir)
	if out, err := goCmd(ctx, opts, 4*time.Minute, args...); err != nil {
		r.Code = b.fn
		testOut := firstLinesRaw(out, err, 40)
		return "test", testOut, fmt.Errorf("teste falhou: %s", firstLines(out, err, 6))
	}
	r.Passed = true
	return "", "", nil
}

// compiled é o resultado de compilar o pacote com a função trocada.
type compiled struct {
	fn      string   // a função como ficou no arquivo (formatada); é o que o compilador viu
	errs    []string // erros com a linha contada dentro de fn
	overlay string   // arquivo de -overlay, para rodar os testes em seguida
}

func compile(ctx context.Context, opts Options, work string, task *Task, code string) (compiled, error) {
	c, _, err := compileWithTrigger(ctx, opts, work, task, code)
	return c, err
}

// compileWithTrigger troca a função da tarefa por code e compila o pacote. O código
// entra por -overlay: nenhum arquivo do projeto é alterado.
func compileWithTrigger(ctx context.Context, opts Options, work string, task *Task, code string) (compiled, string, error) {
	var src []byte
	var err error
	if opts.GoImports {
		src, err = task.ReplaceWithGoImports(code)
	} else {
		src, err = task.Replace(code)
	}
	if err != nil {
		errs := compilerErrors(err.Error(), filepath.Base(task.path), task.Func.LineStart, lineCount(code))
		return compiled{fn: code, errs: errs}, "build", err
	}
	replaced := filepath.Join(work, "replaced.go")
	if err := os.WriteFile(replaced, src, 0o644); err != nil {
		return compiled{}, "build", err
	}
	overlay, _ := json.Marshal(map[string]any{"Replace": map[string]string{task.path: replaced}})
	c := compiled{overlay: filepath.Join(work, "overlay.json")}
	if err := os.WriteFile(c.overlay, overlay, 0o644); err != nil {
		return compiled{}, "build", err
	}
	var start int
	c.fn, start = locate(src, code)
	dir := "./" + filepath.Dir(task.Func.File)
	if out, err := goCmd(ctx, opts, 3*time.Minute, "build", "-overlay="+c.overlay, "-o", os.DevNull, dir); err != nil {
		if c.fn != "" {
			c.errs = compilerErrors(out, filepath.Base(replaced), start, lineCount(c.fn))
		}
		return c, "build", fmt.Errorf("não compila: %s", firstLines(out, err, 4))
	}
	if opts.Vet {
		if out, err := goCmd(ctx, opts, 2*time.Minute, "vet", "-overlay="+c.overlay, dir); err != nil {
			if c.fn != "" {
				c.errs = compilerErrors(out, filepath.Base(replaced), start, lineCount(c.fn))
			}
			return c, "vet", fmt.Errorf("não passou no vet: %s", firstLines(out, err, 4))
		}
	}
	return c, "", nil
}

func firstLinesRaw(out string, err error, n int) string {
	out = strings.TrimSpace(out)
	if out == "" {
		return err.Error()
	}
	lines := strings.Split(out, "\n")
	if len(lines) > n {
		lines = lines[:n]
	}
	return strings.Join(lines, "\n")
}

func firstLines(out string, err error, n int) string {
	out = strings.TrimSpace(out)
	if out == "" {
		return err.Error()
	}
	lines := strings.Split(out, "\n")
	if len(lines) > n {
		lines = lines[:n]
	}
	return strings.Join(lines, " | ")
}

type infraErr struct {
	err error
}

func (e *infraErr) Error() string { return e.err.Error() }
func (e *infraErr) Unwrap() error { return e.err }

// chat pede a função ao modelo, sem restrição de formato.
func chat(ctx context.Context, opts Options, system, prompt string) (string, bool, error) {
	body, _ := json.Marshal(map[string]any{
		"model":  opts.Model,
		"stream": false,
		"think":  false,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": prompt},
		},
		"options": map[string]any{
			"temperature":    0,
			"top_p":          1,
			"repeat_penalty": 1.05,
			"num_ctx":        8192,
			"num_predict":    1000,
			"seed":           42,
		},
	})
	url := strings.TrimRight(opts.Host, "/") + "/api/chat"
	client := http.Client{Timeout: 5 * time.Minute}
	var lastErr error
	for attempt := 0; attempt <= 3; attempt++ {
		if attempt > 0 {
			wait := time.Duration(1<<(attempt-1)) * time.Second
			select {
			case <-ctx.Done():
				return "", false, ctx.Err()
			case <-time.After(wait):
			}
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return "", false, err
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := client.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return "", false, ctx.Err()
			}
			lastErr = &infraErr{err: err}
			continue
		}
		data, rerr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if rerr != nil {
			lastErr = &infraErr{err: rerr}
			continue
		}
		if resp.StatusCode >= 500 {
			lastErr = &infraErr{err: fmt.Errorf("ollama: %s: %s", resp.Status, strings.TrimSpace(string(data)))}
			continue
		}
		if resp.StatusCode != http.StatusOK {
			return "", false, fmt.Errorf("ollama: %s: %s", resp.Status, strings.TrimSpace(string(data)))
		}
		var out struct {
			DoneReason string `json:"done_reason"`
			Message    struct {
				Content string `json:"content"`
			} `json:"message"`
		}
		if err := json.Unmarshal(data, &out); err != nil {
			return "", false, fmt.Errorf("ollama: resposta inválida: %w", err)
		}
		return out.Message.Content, out.DoneReason == "length", nil
	}
	return "", false, lastErr
}
