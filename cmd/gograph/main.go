// Comando gograph: indexa um projeto Go como grafo de funções em SQLite.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/devalexandre/gograph/internal/corpus"
	"github.com/devalexandre/gograph/internal/describe"
	"github.com/devalexandre/gograph/internal/implement"
	"github.com/devalexandre/gograph/internal/indexer"
	"github.com/devalexandre/gograph/internal/server"
	"github.com/devalexandre/gograph/internal/store"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "index":
		err = runIndex(os.Args[2:])
	case "watch":
		err = runWatch(os.Args[2:])
	case "server":
		err = runServer(os.Args[2:])
	case "describe":
		err = runDescribe(os.Args[2:])
	case "corpus":
		err = runCorpus(os.Args[2:])
	case "bench":
		err = runBench(os.Args[2:])
	case "audit":
		err = runAudit(os.Args[2:])
	case "-h", "--help", "help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "comando desconhecido: %s\n\n", os.Args[1])
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `uso: gograph <comando> [opções]

comandos:
  index [opções] [dir]      indexa o projeto Go em dir (padrão: .)
  watch [opções] [dir]      monitora mudanças no código e atualiza o grafo continuamente
  server [opções] [dir]     sobe a interface web (React + Tailwind) para visualizar o grafo
  describe [opções] [dir]   gera as descrições das funções pendentes
  bench [opções] [dir]      mede se um modelo escreve funções que compilam e passam nos testes
  corpus <etapa> [opções]   monta o dataset com repositórios open source
                            (etapas: search, add, fetch, index, describe, implement, status)

rode "gograph <comando> -h" para ver as opções.
`)
}

func runIndex(args []string) error {
	fs := flag.NewFlagSet("index", flag.ExitOnError)
	dbPath := fs.String("db", "", "arquivo SQLite (padrão: <dir>/.gograph/graph.db)")
	tests := fs.Bool("tests", false, "incluir arquivos _test.go")
	chaOnly := fs.Bool("cha", false, "usar só CHA, sem VTA (mais rápido, menos preciso)")
	maxTargets := fs.Int("max-targets", indexer.DefaultMaxTargets, "alvos por chamada dinâmica antes de tratá-la como não resolvida")
	fs.Parse(args)

	dir := "."
	if fs.NArg() > 0 {
		dir = fs.Arg(0)
	}
	dir, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	if *dbPath == "" {
		*dbPath = filepath.Join(dir, ".gograph", "graph.db")
	}

	total := time.Now()
	res, err := indexer.Index(indexer.Options{Dir: dir, Tests: *tests, CHAOnly: *chaOnly, MaxTargets: *maxTargets})
	if err != nil {
		return err
	}

	st, err := store.Open(*dbPath)
	if err != nil {
		return err
	}
	defer st.Close()

	writeStart := time.Now()
	meta := map[string]string{"module": res.Module, "last_commit": gitHead(dir), "root": dir}
	stats, err := st.ReplaceGraph(res.Functions, res.Edges, meta)
	if err != nil {
		return err
	}
	write := time.Since(writeStart)

	for _, sk := range res.Skipped {
		fmt.Fprintf(os.Stderr, "aviso: pacote ignorado %s: %s\n", sk.Path, sk.Reason)
	}
	t := res.Timings
	fmt.Printf("módulo:    %s\n", res.Module)
	fmt.Printf("funções:   %d (%d novas, %d alteradas, %d removidas)\n", stats.Functions, stats.Added, stats.Changed, stats.Removed)
	if n := len(res.Skipped); n > 0 {
		fmt.Printf("ignorados: %d pacotes que não compilam\n", n)
	}
	fmt.Printf("externas:  %d\n", stats.External)
	fmt.Printf("arestas:   %d\n", stats.Edges)
	fmt.Printf("tempo:     %s (carga %s, ssa %s, call graph %s, extração %s, sqlite %s)\n",
		round(time.Since(total)), round(t.Load), round(t.SSA), round(t.CallGraph), round(t.Extract), round(write))
	fmt.Printf("banco:     %s\n", *dbPath)
	return nil
}

func round(d time.Duration) time.Duration { return d.Round(10 * time.Millisecond) }

// gitHead devolve o commit atual do projeto, ou vazio fora de um repositório.
func gitHead(dir string) string {
	out, err := exec.Command("git", "-C", dir, "rev-parse", "HEAD").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// backendFlags são as opções de modelo comuns a describe e corpus describe.
type backendFlags struct {
	backend, model, effort, host string
	workers                      int
}

func addBackendFlags(fs *flag.FlagSet, defaultBackend string) *backendFlags {
	bf := &backendFlags{}
	fs.StringVar(&bf.backend, "backend", defaultBackend, "ollama ou anthropic")
	fs.StringVar(&bf.model, "model", "", "modelo (padrão: qwen3.5:4b no ollama, claude-opus-5-5 na anthropic)")
	fs.StringVar(&bf.effort, "effort", "low", "esforço do modelo na anthropic: low, medium, high, xhigh ou max")
	fs.StringVar(&bf.host, "ollama-host", "http://localhost:11434", "endereço do Ollama")
	fs.IntVar(&bf.workers, "j", 0, "chamadas simultâneas (padrão: 1 no ollama, 4 na anthropic)")
	return bf
}

func (bf *backendFlags) build() (describe.Backend, error) {
	switch bf.backend {
	case "ollama":
		if bf.model == "" {
			bf.model = "qwen3.5:4b"
		}
		if bf.workers == 0 {
			bf.workers = 1
		}
		return describe.NewOllama(bf.host, bf.model), nil
	case "anthropic":
		if os.Getenv("ANTHROPIC_API_KEY") == "" && os.Getenv("ANTHROPIC_AUTH_TOKEN") == "" {
			return nil, fmt.Errorf("defina ANTHROPIC_API_KEY para usar o backend anthropic")
		}
		if bf.model == "" {
			bf.model = "claude-opus-5-5"
		}
		if bf.workers == 0 {
			bf.workers = 4
		}
		return describe.NewAnthropic(bf.model, bf.effort), nil
	}
	return nil, fmt.Errorf("backend desconhecido: %s", bf.backend)
}

// printUsage mostra os tokens consumidos, quando o backend informa.
func printUsage(be describe.Backend, described int) {
	u, ok := be.(interface{ Usage() (int64, int64) })
	if !ok || described == 0 {
		return
	}
	in, out := u.Usage()
	fmt.Printf("tokens:    %d de entrada, %d de saída (média %d + %d por descrição)\n",
		in, out, in/int64(described), out/int64(described))
}

func progressLine(done, total int, name string, err error) {
	status := "ok"
	if err != nil {
		status = "FALHOU: " + err.Error()
	}
	fmt.Printf("[%d/%d] %s %s\n", done, total, name, status)
}

func runDescribe(args []string) error {
	fs := flag.NewFlagSet("describe", flag.ExitOnError)
	dbPath := fs.String("db", "", "arquivo SQLite (padrão: <dir>/.gograph/graph.db)")
	bf := addBackendFlags(fs, "ollama")
	limit := fs.Int("limit", 0, "máximo de funções nesta rodada (0 = todas)")
	match := fs.String("match", "", "só funções cujo nome contém este texto")
	maxLines := fs.Int("max-lines", 400, "pula funções com mais linhas que isso (0 = sem limite)")
	jsonl := fs.String("jsonl", "", "acrescenta entrada e saída de cada descrição neste arquivo JSONL (dataset)")
	fs.Parse(args)

	dir := "."
	if fs.NArg() > 0 {
		dir = fs.Arg(0)
	}
	dir, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	if *dbPath == "" {
		*dbPath = filepath.Join(dir, ".gograph", "graph.db")
	}
	if _, err := os.Stat(*dbPath); err != nil {
		return fmt.Errorf("índice não encontrado em %s; rode gograph index primeiro", *dbPath)
	}
	st, err := store.Open(*dbPath)
	if err != nil {
		return err
	}
	defer st.Close()

	be, err := bf.build()
	if err != nil {
		return err
	}
	workers := &bf.workers

	opts := describe.Options{
		Root: dir, Backend: be, Workers: *workers, Limit: *limit, Match: *match, MaxLines: *maxLines,
		Repair:   bf.backend == "ollama",
		Progress: progressLine,
	}
	if *jsonl != "" {
		f, err := os.OpenFile(*jsonl, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			return err
		}
		defer f.Close()
		opts.Dataset = f
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	rep, err := describe.Run(ctx, st, opts)
	if rep != nil {
		fmt.Printf("\nmodelo:     %s\n", be.Name())
		fmt.Printf("descritas:  %d\n", rep.Described)
		fmt.Printf("falharam:   %d\n", rep.Failed)
		if rep.TooLong > 0 {
			fmt.Printf("puladas:    %d com mais de %d linhas\n", rep.TooLong, *maxLines)
		}
		if rep.Described > 0 {
			fmt.Printf("tempo:      %s (%s por descrição)\n", round(rep.Elapsed), round(rep.Elapsed/time.Duration(rep.Described)))
		}
		printUsage(be, rep.Described+rep.Failed)
	}
	return err
}

func runCorpus(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("uso: gograph corpus <search|add|fetch|index|describe|implement|teach|recontext|audit|status> [opções]")
	}
	step, args := args[0], args[1:]
	fs := flag.NewFlagSet("corpus "+step, flag.ExitOnError)
	dir := fs.String("dir", "corpus", "pasta do corpus")
	count := fs.Int("n", 50, "search: quantos repositórios acrescentar")
	minStars := fs.Int("min-stars", 1000, "search: mínimo de estrelas")
	maxSizeMB := fs.Int("max-size-mb", 80, "search: tamanho máximo do repositório")
	timeout := fs.Duration("timeout", 5*time.Minute, "index: tempo máximo por repositório")
	sampleN := fs.Int("sample", 150, "describe: funções sorteadas por repositório (os callees delas entram a mais)")
	minLines := fs.Int("min-lines", 5, "describe: tamanho mínimo das funções sorteadas")
	maxLines := fs.Int("max-lines", 300, "describe: pula funções com mais linhas que isso")
	maxRepos := fs.Int("repos", 0, "describe: máximo de repositórios nesta rodada (0 = todos)")
	gobin := fs.String("go", "", "teach: binário do Go usado para compilar (padrão: go do PATH)")
	ctxMode := fs.String("ctx", implement.CtxReal, "modo de contexto: real, oracle-distract ou oracle")
	budget := fs.Int("budget", implement.DefaultBudgetChars, "orçamento máximo de caracteres do contexto em ctx=real")
	bf := addBackendFlags(fs, "anthropic")
	fs.Parse(args)
	implement.DefaultCtxMode = *ctxMode
	implement.DefaultBudgetChars = *budget

	c, err := corpus.Open(*dir)
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	show := func(r *corpus.Repo) {
		line := fmt.Sprintf("%-10s %-45s %s", r.Status, r.Name, r.License)
		if r.Functions > 0 {
			line += fmt.Sprintf("  %d funções", r.Functions)
		}
		if r.Described > 0 {
			line += fmt.Sprintf("  %d descritas", r.Described)
		}
		if r.Error != "" {
			line += "  (" + r.Error + ")"
		}
		fmt.Println(line)
	}

	switch step {
	case "teach":
		return runTeach(ctx, c, bf, *sampleN, *minLines, *maxLines, *maxRepos, *gobin)
	case "recontext":
		return runRecontext(c)
	case "audit":
		// Para cada função da massa de escrita, o que o corpo usa do próprio
		// projeto e a entrada não mostra. Não chama modelo; serve para filtrar
		// a massa antes de treinar.
		path := filepath.Join(c.Dir, "audit.jsonl")
		out, err := os.Create(path)
		if err != nil {
			return err
		}
		defer out.Close()
		if *gobin != "" {
			os.Setenv("PATH", filepath.Dir(*gobin)+string(os.PathListSeparator)+os.Getenv("PATH"))
		}
		env := c.GoEnv()
		if *gobin != "" {
			env = append(env, "GOROOT="+filepath.Dir(filepath.Dir(*gobin)))
		}
		total, clean, failed := 0, 0, 0
		for _, r := range c.Repos {
			if ctx.Err() != nil {
				break
			}
			if r.Status != corpus.StatusIndexed && r.Status != corpus.StatusDescribed {
				continue
			}
			st, err := store.Open(c.DBPath(r))
			if err != nil {
				return err
			}
			rctx, cancel := context.WithTimeout(ctx, *timeout)
			rows, err := implement.Audit(rctx, st, implement.AuditOptions{
				Root: c.SrcDir(r), N: *sampleN, MinLines: *minLines, MaxLines: *maxLines, Env: env, Out: out,
			})
			cancel()
			st.Close()
			if err != nil {
				failed++
				fmt.Printf("FALHOU   %-45s %v\n", r.Name, firstLine(err.Error()))
				continue
			}
			n := 0
			for _, row := range rows {
				if row.Missing == 0 {
					n++
				}
			}
			total, clean = total+len(rows), clean+n
			fmt.Printf("%4d/%-4d %s\n", n, len(rows), r.Name)
		}
		fmt.Printf("%d funções auditadas, %d com tudo do projeto à vista; %d repositórios falharam; resultado em %s\n", total, clean, failed, path)
		return nil
	case "implement":
		// Dataset da tarefa de escrita: sai do próprio código, sem modelo professor.
		path := filepath.Join(c.Dir, "implement.jsonl")
		out, err := os.Create(path)
		if err != nil {
			return err
		}
		defer out.Close()
		total := 0
		for _, r := range c.Repos {
			if r.Status != corpus.StatusIndexed && r.Status != corpus.StatusDescribed {
				continue
			}
			st, err := store.Open(c.DBPath(r))
			if err != nil {
				return err
			}
			n, err := implement.Export(st, c.SrcDir(r), r.Name, r.License, *sampleN, *minLines, *maxLines, out)
			st.Close()
			if err != nil {
				return err
			}
			total += n
		}
		fmt.Printf("%d exemplos em %s\n", total, path)
		return nil
	case "search":
		n, err := c.Search(ctx, corpus.SearchOptions{Count: *count, MinStars: *minStars, MinSizeKB: 200, MaxSizeKB: *maxSizeMB * 1024})
		fmt.Printf("%d repositórios acrescentados; %d no manifesto\n", n, len(c.Repos))
		return err
	case "add":
		for _, name := range fs.Args() {
			if err := c.Add(ctx, name); err != nil {
				return err
			}
		}
		fmt.Printf("%d repositórios no manifesto\n", len(c.Repos))
		return nil
	case "fetch":
		return c.Fetch(ctx, show)
	case "index":
		return c.Index(ctx, *timeout, show)
	case "status":
		byStatus := map[string]int{}
		funcs, described := 0, 0
		for _, r := range c.Repos {
			show(r)
			byStatus[r.Status]++
			funcs += r.Functions
			described += r.Described
		}
		fmt.Printf("\n%d repositórios %v, %d funções indexadas, %d descritas\n", len(c.Repos), byStatus, funcs, described)
		return nil
	case "describe":
		be, err := bf.build()
		if err != nil {
			return err
		}
		out, err := os.OpenFile(c.DatasetPath(), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			return err
		}
		defer out.Close()
		start := time.Now()
		total, failed, repos := 0, 0, 0
		for _, r := range c.Repos {
			if r.Status != corpus.StatusIndexed || ctx.Err() != nil || (*maxRepos > 0 && repos >= *maxRepos) {
				continue
			}
			repos++
			fmt.Printf("== %s (%s)\n", r.Name, r.License)
			st, err := store.Open(c.DBPath(r))
			if err != nil {
				return err
			}
			rep, err := describe.Run(ctx, st, describe.Options{
				Root: c.SrcDir(r), Backend: be, Workers: bf.workers,
				Sample: *sampleN, MinLines: *minLines, MaxLines: *maxLines,
				Source: r.Name, License: r.License, Dataset: out, Progress: progressLine,
			})
			st.Close()
			if rep != nil {
				r.Described += rep.Described
				total += rep.Described
				failed += rep.Failed
			}
			if err == nil {
				r.Status = corpus.StatusDescribed
			}
			if serr := c.Save(); serr != nil {
				return serr
			}
			if err != nil {
				return err
			}
		}
		fmt.Printf("\nmodelo:    %s\ndescritas: %d em %d repositórios (%d falharam)\n", be.Name(), total, repos, failed)
		if total > 0 {
			fmt.Printf("tempo:     %s (%s por descrição)\ndataset:   %s\n", round(time.Since(start)), round(time.Since(start)/time.Duration(total)), c.DatasetPath())
		}
		printUsage(be, total+failed)
		return nil
	}
	return fmt.Errorf("etapa desconhecida: %s", step)
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	if len(s) > 160 {
		s = s[:160]
	}
	return s
}

// runRecontext refaz o contexto das entradas de <corpus>/teach.jsonl com o
// formato atual de implement.BuildTask, sem chamar o modelo professor. O
// arquivo anterior fica em teach.jsonl.bak.
func runRecontext(c *corpus.Corpus) error {
	path := filepath.Join(c.Dir, "teach.jsonl")
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	repos := map[string]*corpus.Repo{}
	for _, r := range c.Repos {
		repos[r.Name] = r
	}
	stores := map[string]*store.Store{}
	defer func() {
		for _, st := range stores {
			st.Close()
		}
	}()
	var out bytes.Buffer
	done, failed := 0, 0
	for _, line := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var r implement.Record
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			failed++ // linha cortada por uma rodada interrompida
			continue
		}
		repo := repos[r.Source]
		if repo == nil {
			failed++
			continue
		}
		st := stores[r.Source]
		if st == nil {
			if st, err = store.Open(c.DBPath(repo)); err != nil {
				return err
			}
			stores[r.Source] = st
		}
		if err := implement.Recontext(st, c.SrcDir(repo), &r); err != nil {
			failed++
			continue
		}
		b, err := json.Marshal(r)
		if err != nil {
			return err
		}
		out.Write(b)
		out.WriteByte('\n')
		done++
	}
	if err := os.WriteFile(path+".bak", data, 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(path, out.Bytes(), 0o644); err != nil {
		return err
	}
	fmt.Printf("%d exemplos refeitos, %d descartados; arquivo anterior em %s.bak\n", done, failed, path)
	return nil
}

// runTeach gera, com o modelo professor, os exemplos de escrita por pedido,
// de alteração e de correção. Grava em <corpus>/teach.jsonl e pode ser
// interrompido: a rodada seguinte pula as funções já feitas.
func runTeach(ctx context.Context, c *corpus.Corpus, bf *backendFlags, sample, minLines, maxLines, maxRepos int, gobin string) error {
	if bf.backend != "anthropic" {
		return fmt.Errorf("corpus teach só funciona com -backend anthropic")
	}
	if _, err := bf.build(); err != nil {
		return err
	}
	be := implement.NewTeacher(bf.model, bf.effort)
	path := filepath.Join(c.Dir, "teach.jsonl")
	done := map[string]bool{}
	if data, err := os.ReadFile(path); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			var r struct {
				Hash string `json:"hash"`
			}
			if json.Unmarshal([]byte(line), &r) == nil && r.Hash != "" {
				done[r.Hash] = true
			}
		}
	}
	out, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer out.Close()
	// Uma rodada interrompida no meio de uma linha deixa o arquivo sem a
	// quebra final: sem isto o próximo exemplo ficaria colado no pedaço.
	if data, err := os.ReadFile(path); err == nil && len(data) > 0 && data[len(data)-1] != '\n' {
		if _, err := out.WriteString("\n"); err != nil {
			return err
		}
	}
	env := c.GoEnv()
	if gobin != "" {
		env = append(env, "GOROOT="+filepath.Dir(filepath.Dir(gobin)))
	}
	start := time.Now()
	var total implement.TeachReport
	repos := 0
	for _, r := range c.Repos {
		if (r.Status != corpus.StatusIndexed && r.Status != corpus.StatusDescribed) || ctx.Err() != nil || (maxRepos > 0 && repos >= maxRepos) {
			continue
		}
		repos++
		fmt.Printf("== %s (%s)\n", r.Name, r.License)
		st, err := store.Open(c.DBPath(r))
		if err != nil {
			return err
		}
		rep, err := implement.Teach(ctx, st, implement.TeachOptions{
			Root: c.SrcDir(r), Source: r.Name, License: r.License, Backend: be, Workers: bf.workers,
			N: sample, MinLines: minLines, MaxLines: maxLines, Go: gobin, Env: env, Done: done, Out: out,
			Progress: func(name string, kinds []string, err error) {
				status := strings.Join(kinds, "+")
				if err != nil {
					status = "FALHOU: " + err.Error()
				}
				fmt.Printf("%-16s %s\n", status, name)
			},
		})
		st.Close()
		if rep != nil {
			total.Functions += rep.Functions
			total.Failed += rep.Failed
			total.Create += rep.Create
			total.Edit += rep.Edit
			total.Fix += rep.Fix
		}
		if err != nil {
			break
		}
	}
	fmt.Printf("\nmodelo:    %s\nfunções:   %d em %d repositórios (%d falharam)\n", be.Name(), total.Functions, repos, total.Failed)
	fmt.Printf("exemplos:  %d de escrita por pedido, %d de alteração, %d de correção\n", total.Create, total.Edit, total.Fix)
	fmt.Printf("tempo:     %s\narquivo:   %s\n", round(time.Since(start)), path)
	printUsage(be, total.Functions)
	return ctx.Err()
}

// runBench mede a escrita de funções: o modelo recebe assinatura e contexto,
// e o resultado é conferido com go build e go test.
// readAnswers lê um JSONL de respostas geradas fora do Ollama. A última
// resposta de cada hash vale.
func readAnswers(path string) (map[string]string, map[string]int64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	answers := map[string]string{}
	timings := map[string]int64{}
	for i, line := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var r struct {
			Hash      string `json:"hash"`
			Answer    string `json:"answer"`
			ElapsedMs int64  `json:"elapsed_ms"`
		}
		if err := json.Unmarshal([]byte(line), &r); err != nil || r.Hash == "" {
			return nil, nil, fmt.Errorf("%s: linha %d sem hash ou inválida", path, i+1)
		}
		answers[r.Hash] = r.Answer
		timings[r.Hash] = r.ElapsedMs
	}
	return answers, timings, nil
}

// runAudit confere se a entrada que o modelo recebe mostra o que o corpo
// original de cada função usa. Não chama modelo nenhum.
func runAudit(args []string) error {
	fs := flag.NewFlagSet("audit", flag.ExitOnError)
	dbPath := fs.String("db", "", "banco do grafo (padrão: <dir>/.gograph/graph.db)")
	n := fs.Int("n", 50, "funções a auditar (as mesmas que o bench mede)")
	minLines := fs.Int("min-lines", 5, "tamanho mínimo da função")
	maxLines := fs.Int("max-lines", 40, "tamanho máximo da função")
	corpusDir := fs.String("corpus", "", "usa o ambiente Go deste corpus")
	outPath := fs.String("out", "", "grava a auditoria de cada função neste JSONL")
	gobin := fs.String("go", "", "binário do Go a usar (padrão: go do PATH)")
	tail := fs.Bool("tail", false, "usa a outra ponta da amostra")
	ctxMode := fs.String("ctx", implement.CtxReal, "modo de contexto: real, oracle-distract ou oracle")
	budget := fs.Int("budget", implement.DefaultBudgetChars, "orçamento máximo de caracteres do contexto em ctx=real")
	fs.Parse(args)
	implement.DefaultCtxMode = *ctxMode
	implement.DefaultBudgetChars = *budget

	dir := "."
	if fs.NArg() > 0 {
		dir = fs.Arg(0)
	}
	dir, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	if *dbPath == "" {
		*dbPath = filepath.Join(dir, ".gograph", "graph.db")
	}
	st, err := store.Open(*dbPath)
	if err != nil {
		return err
	}
	defer st.Close()
	opts := implement.AuditOptions{Root: dir, N: *n, MinLines: *minLines, MaxLines: *maxLines, Tail: *tail}
	if *corpusDir != "" {
		opts.Env = (&corpus.Corpus{Dir: *corpusDir}).GoEnv()
	}
	if *gobin != "" {
		if opts.Env == nil {
			opts.Env = os.Environ()
		}
		// O carregador de pacotes procura o "go" no PATH deste processo.
		path := filepath.Dir(*gobin) + string(os.PathListSeparator) + os.Getenv("PATH")
		os.Setenv("PATH", path)
		opts.Env = append(opts.Env, "GOROOT="+filepath.Dir(filepath.Dir(*gobin)), "PATH="+path)
	}
	if *outPath != "" {
		f, err := os.Create(*outPath)
		if err != nil {
			return err
		}
		defer f.Close()
		opts.Out = f
	}
	rows, err := implement.Audit(context.Background(), st, opts)
	if err != nil {
		return err
	}
	type key struct{ origin, kind string }
	used, missing := map[key]int{}, map[key]int{}
	clean := 0
	for _, r := range rows {
		if r.Missing == 0 {
			clean++
		}
		for _, s := range r.Symbols {
			k := key{s.Origin, s.Kind}
			used[k]++
			if !s.Shown {
				missing[k]++
			}
		}
	}
	fmt.Printf("%d funções; %d com tudo o que usam do projeto à vista na entrada\n", len(rows), clean)
	for _, o := range []string{"pacote", "projeto", "externo", "stdlib"} {
		for _, k := range []string{"função", "método", "campo", "tipo", "constante", "variável"} {
			if u := used[key{o, k}]; u > 0 {
				fmt.Printf("  %-8s %-10s usados %4d  fora da entrada %4d\n", o, k, u, missing[key{o, k}])
			}
		}
	}
	return nil
}

func runBench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	dbPath := fs.String("db", "", "banco do grafo (padrão: <dir>/.gograph/graph.db)")
	model := fs.String("model", "qwen2.5-coder:1.5b", "modelo no Ollama")
	host := fs.String("ollama-host", "http://localhost:11434", "endereço do Ollama")
	n := fs.Int("n", 50, "funções a testar")
	minLines := fs.Int("min-lines", 5, "tamanho mínimo da função")
	maxLines := fs.Int("max-lines", 40, "tamanho máximo da função")
	tests := fs.Bool("tests", false, "roda go test nas funções cobertas por teste (executa código gerado pelo modelo)")
	corpusDir := fs.String("corpus", "", "usa o ambiente Go deste corpus (cache de módulos isolado)")
	outPath := fs.String("out", "", "grava cada tentativa neste JSONL")
	gobin := fs.String("go", "", "binário do Go a usar (padrão: go do PATH)")
	retries := fs.Int("retries", 0, "vezes que o modelo recebe o erro do compilador para corrigir")
	dump := fs.Bool("dump", false, "só grava em -out as tarefas (entrada de cada função), sem chamar modelo")
	answersPath := fs.String("answers", "", "mede as respostas deste JSONL ({hash, answer}) em vez de chamar o Ollama")
	tail := fs.Bool("tail", false, "usa a outra ponta da amostra: funções que a medição normal não pega")
	goimports := fs.Bool("goimports", true, "aplica goimports ao arquivo antes do go build")
	vet := fs.Bool("vet", false, "roda go vet após o go build")
	workers := fs.Int("workers", 1, "workers em paralelo")
	ctxMode := fs.String("ctx", implement.CtxReal, "modo de contexto: real, oracle-distract ou oracle")
	budget := fs.Int("budget", implement.DefaultBudgetChars, "orçamento máximo de caracteres do contexto em ctx=real")
	fs.Parse(args)
	implement.DefaultCtxMode = *ctxMode
	implement.DefaultBudgetChars = *budget

	dir := "."
	if fs.NArg() > 0 {
		dir = fs.Arg(0)
	}
	dir, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	if *dbPath == "" {
		*dbPath = filepath.Join(dir, ".gograph", "graph.db")
	}
	st, err := store.Open(*dbPath)
	if err != nil {
		return err
	}
	defer st.Close()

	opts := implement.Options{
		Root: dir, Host: *host, Model: *model, N: *n, MinLines: *minLines, MaxLines: *maxLines, Tests: *tests, Go: *gobin, Retries: *retries,
		Dump: *dump, Tail: *tail, GoImports: *goimports, Vet: *vet, Workers: *workers,
		Progress: func(r implement.Result) {
			status := "não compila"
			switch {
			case r.Passed:
				status = "compila, testes passam"
			case r.Built && r.Covered:
				status = "compila, testes falham"
			case r.Built:
				status = "compila"
			}
			fmt.Printf("%-24s %s\n", status, r.Name)
		},
	}
	if *dump && *outPath == "" {
		return fmt.Errorf("-dump precisa de -out")
	}
	if *answersPath != "" {
		opts.Answers, opts.AnswerTimings, err = readAnswers(*answersPath)
		if err != nil {
			return err
		}
		if *retries == 0 {
			*model = "respostas de " + filepath.Base(*answersPath)
		}
	}
	if *corpusDir != "" {
		opts.Env = (&corpus.Corpus{Dir: *corpusDir}).GoEnv()
		if *goimports {
			modcache, _ := filepath.Abs(filepath.Join(*corpusDir, "modcache"))
			os.Setenv("GOMODCACHE", modcache)
		}
	}
	if *gobin != "" {
		// O gvm exporta GOROOT da versão ativa; outro binário precisa do seu.
		if opts.Env == nil {
			opts.Env = os.Environ()
		}
		opts.Env = append(opts.Env, "GOROOT="+filepath.Dir(filepath.Dir(*gobin)))
	}
	if *outPath != "" {
		if *dump {
			f, err := os.Create(*outPath)
			if err != nil {
				return err
			}
			defer f.Close()
			opts.Out = f
		} else {
			if prev, _, rerr := readAnswers(*outPath); rerr == nil && len(prev) > 0 {
				opts.SkipHashes = map[string]bool{}
				for h := range prev {
					opts.SkipHashes[h] = true
				}
				fmt.Printf("retomando %s (%d funções já gravadas)\n", *outPath, len(opts.SkipHashes))
			}
			f, err := os.OpenFile(*outPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
			if err != nil {
				return err
			}
			defer f.Close()
			opts.Out = f
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	rep, err := implement.Run(ctx, st, opts)
	if rep != nil && *dump {
		fmt.Printf("%d tarefas em %s\n", rep.Total, *outPath)
		return err
	}
	if rep != nil {
		fmt.Printf("\nmodelo:   %s\n", *model)
		fmt.Printf("compilam: %d de %d\n", rep.Built, rep.Total)
		if *retries > 0 {
			fmt.Printf("          %d delas só depois de receber o erro do compilador\n", rep.Fixed)
		}
		if rep.Skipped > 0 {
			fmt.Printf("puladas:  %d (pacote não compila no original)\n", rep.Skipped)
		}
		if *tests {
			fmt.Printf("testes:   %d de %d funções cobertas por teste\n", rep.Passed, rep.Covered)
		}
		fmt.Printf("tempo:    %s\n", round(rep.Elapsed))
	}
	return err
}

func runWatch(args []string) error {
	fs := flag.NewFlagSet("watch", flag.ExitOnError)
	dbPath := fs.String("db", "", "arquivo SQLite (padrão: <dir>/.gograph/graph.db)")
	tests := fs.Bool("tests", false, "incluir arquivos _test.go")
	chaOnly := fs.Bool("cha", false, "usar só CHA, sem VTA")
	interval := fs.Duration("interval", time.Second, "intervalo de verificação de mudanças")
	fs.Parse(args)

	dir := "."
	if fs.NArg() > 0 {
		dir = fs.Arg(0)
	}
	dir, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	if *dbPath == "" {
		*dbPath = filepath.Join(dir, ".gograph", "graph.db")
	}

	reindex := func() {
		res, err := indexer.Index(indexer.Options{Dir: dir, Tests: *tests, CHAOnly: *chaOnly})
		if err != nil {
			fmt.Fprintf(os.Stderr, "[watch] erro ao indexar: %v\n", err)
			return
		}
		st, err := store.Open(*dbPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[watch] erro ao abrir banco: %v\n", err)
			return
		}
		defer st.Close()
		meta := map[string]string{"module": res.Module, "last_commit": gitHead(dir), "root": dir}
		stats, err := st.ReplaceGraph(res.Functions, res.Edges, meta)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[watch] erro ao gravar grafo: %v\n", err)
			return
		}
		fmt.Printf("[watch %s] grafo atualizado: %d funções (+%d ~%d -%d), %d arestas\n",
			time.Now().Format("15:04:05"), stats.Functions, stats.Added, stats.Changed, stats.Removed, stats.Edges)
	}

	fmt.Printf("monitorando %s -> %s (Ctrl+C para sair)\n", dir, *dbPath)
	reindex()
	prev := snapshotGoFiles(dir)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ticker := time.NewTicker(*interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			cur := snapshotGoFiles(dir)
			if !mapsEqual(prev, cur) {
				prev = cur
				reindex()
			}
		}
	}
}

func snapshotGoFiles(dir string) map[string]int64 {
	snap := map[string]int64{}
	_ = filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		name := d.Name()
		if d.IsDir() {
			if name == ".git" || name == ".gograph" || name == "vendor" || name == "node_modules" {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(name, ".go") || name == "go.mod" {
			if info, err := d.Info(); err == nil {
				snap[p] = info.ModTime().UnixNano() ^ info.Size()
			}
		}
		return nil
	})
	return snap
}

func mapsEqual(a, b map[string]int64) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if bv, ok := b[k]; !ok || bv != v {
			return false
		}
	}
	return true
}

func runServer(args []string) error {
	fs := flag.NewFlagSet("server", flag.ExitOnError)
	dbPath := fs.String("db", "", "arquivo SQLite (padrão: <dir>/.gograph/graph.db)")
	addr := fs.String("addr", ":8080", "endereço HTTP do servidor visual")
	fs.Parse(args)

	dir := "."
	if fs.NArg() > 0 {
		dir = fs.Arg(0)
	}
	dir, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	if *dbPath == "" {
		*dbPath = filepath.Join(dir, ".gograph", "graph.db")
	}
	st, err := store.Open(*dbPath)
	if err != nil {
		return err
	}
	defer st.Close()

	h, err := server.Handler(st, dir)
	if err != nil {
		return err
	}
	fmt.Printf("gograph server em http://localhost%s (banco: %s)\n", *addr, *dbPath)
	return http.ListenAndServe(*addr, h)
}
