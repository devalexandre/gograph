package describe

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/devalexandre/gograph/internal/store"
)

// Options configura uma rodada de geração.
type Options struct {
	Root     string // raiz do projeto, para ler o código
	Backend  Backend
	Workers  int    // chamadas simultâneas ao modelo
	Limit    int    // máximo de funções nesta rodada; 0 = todas
	Match    string // só funções cujo nome contém este texto
	MaxLines int    // funções maiores que isso são puladas; 0 = sem limite
	// Sample, se maior que zero, descreve só uma amostra: Sample funções com
	// pelo menos MinLines linhas, fora de arquivos gerados, mais as funções
	// que elas chamam (para que recebam o purpose dos callees).
	Sample   int
	MinLines int
	Source   string // origem do código (repositório), gravada no dataset
	License  string

	// Repair corrige saídas quase válidas em vez de rejeitá-las (ver Repair).
	// Fica desligado na geração do dataset, onde a rejeição pede nova
	// tentativa ao modelo professor.
	Repair bool

	Dataset  io.Writer // se não for nil, recebe um JSONL com entrada e saída
	Progress func(done, total int, name string, err error)
}

// Report resume uma rodada.
type Report struct {
	Described int
	Failed    int
	TooLong   int // puladas por passarem de MaxLines
	Elapsed   time.Duration
	Failures  []string
}

// Record é uma linha do dataset: a entrada do modelo e a descrição aceita.
type Record struct {
	Source  string `json:"source,omitempty"`
	License string `json:"license,omitempty"`

	Name   string          `json:"name"`
	Hash   string          `json:"hash"`
	Model  string          `json:"model"`
	System string          `json:"system"`
	Input  string          `json:"input"`
	Output json.RawMessage `json:"output"`
}

// Run gera as descrições pendentes, das funções chamadas para as que chamam,
// para que cada função já receba o purpose dos seus callees.
func Run(ctx context.Context, st *store.Store, opts Options) (*Report, error) {
	start := time.Now()
	if opts.Workers < 1 {
		opts.Workers = 1
	}
	pending, err := st.PendingDescriptions()
	if err != nil {
		return nil, err
	}
	rep := &Report{}
	byID := map[int64]store.FuncRow{}
	for _, f := range pending {
		if opts.Match != "" && !strings.Contains(f.Name, opts.Match) {
			continue
		}
		if opts.MaxLines > 0 && f.LineEnd-f.LineStart+1 > opts.MaxLines {
			rep.TooLong++
			continue
		}
		byID[f.ID] = f
	}
	if opts.Sample > 0 {
		byID, err = sample(st, opts, byID)
		if err != nil {
			return nil, err
		}
	}

	// deps[id]: callees pendentes de id. users[id]: pendentes que chamam id.
	deps := map[int64]map[int64]bool{}
	users := map[int64][]int64{}
	for id := range byID {
		deps[id] = map[int64]bool{}
		callees, err := st.Callees(id)
		if err != nil {
			return nil, err
		}
		for _, c := range callees {
			if _, ok := byID[c.ID]; ok && c.ID != id {
				deps[id][c.ID] = true
				users[c.ID] = append(users[c.ID], id)
			}
		}
	}

	total := len(byID)
	if opts.Limit > 0 && opts.Limit < total {
		total = opts.Limit
	}
	var mu sync.Mutex // protege rep, Dataset e o contador
	done := 0

	// Fila contínua: assim que uma função termina, as que dependiam dela
	// ficam livres e ocupam o worker, sem esperar o resto do lote.
	results := make(chan int64)
	inflight, started := 0, 0
	for {
		for inflight < opts.Workers && started < total && ctx.Err() == nil {
			id, ok := nextReady(deps, byID, inflight == 0)
			if !ok {
				break
			}
			delete(deps, id)
			inflight++
			started++
			go func(f store.FuncRow) {
				rec, err := describeOne(ctx, st, opts, f)
				mu.Lock()
				done++
				if err != nil {
					rep.Failed++
					rep.Failures = append(rep.Failures, fmt.Sprintf("%s: %v", f.Name, err))
				} else {
					rep.Described++
					if opts.Dataset != nil {
						line, _ := json.Marshal(rec)
						opts.Dataset.Write(append(line, '\n'))
					}
				}
				if opts.Progress != nil {
					opts.Progress(done, total, f.Name, err)
				}
				mu.Unlock()
				results <- f.ID
			}(byID[id])
		}
		if inflight == 0 {
			break
		}
		id := <-results
		inflight--
		for _, u := range users[id] {
			if d, ok := deps[u]; ok {
				delete(d, id)
			}
		}
	}
	rep.Elapsed = time.Since(start)
	if rep.Described > 0 {
		if err := st.SetMeta("model_version", opts.Backend.Name()); err != nil {
			return rep, err
		}
	}
	return rep, ctx.Err()
}

// nextReady escolhe a próxima função cujos callees já foram descritos. Em um
// ciclo de chamadas nenhuma fica livre; com breakCycle sai a que depende de
// menos funções.
func nextReady(deps map[int64]map[int64]bool, byID map[int64]store.FuncRow, breakCycle bool) (int64, bool) {
	ready, best := int64(-1), int64(-1)
	bestN := 0
	for id, d := range deps {
		if len(d) == 0 {
			if ready < 0 || byID[id].Name < byID[ready].Name {
				ready = id
			}
			continue
		}
		if best < 0 || len(d) < bestN || (len(d) == bestN && byID[id].Name < byID[best].Name) {
			best, bestN = id, len(d)
		}
	}
	if ready >= 0 {
		return ready, true
	}
	if breakCycle && best >= 0 {
		return best, true
	}
	return 0, false
}

// describeOne gera, valida e grava a descrição de uma função. Uma saída fora
// do formato ganha uma segunda tentativa com o motivo da rejeição.
func describeOne(ctx context.Context, st *store.Store, opts Options, f store.FuncRow) (*Record, error) {
	callees, err := st.Callees(f.ID)
	if err != nil {
		return nil, err
	}
	in, err := BuildInput(opts.Root, f, callees)
	if err != nil {
		return nil, err
	}

	user := in.Text
	var desc *Description
	for attempt := 1; ; attempt++ {
		raw, err := opts.Backend.Generate(ctx, SystemPrompt, user)
		if err != nil {
			return nil, err
		}
		if opts.Repair {
			desc, err = Repair(raw, in.Allowed)
		} else {
			desc, err = Parse(raw, in.Allowed)
		}
		if err == nil {
			break
		}
		if attempt == 2 {
			return nil, fmt.Errorf("saída fora do formato: %w", err)
		}
		user = in.Text + "\nUma resposta anterior foi rejeitada: " + err.Error() + ". Gere de novo respeitando o formato.\n"
	}

	out, err := json.Marshal(desc)
	if err != nil {
		return nil, err
	}
	if err := st.SetDescription(f.ID, string(out), f.Hash); err != nil {
		return nil, err
	}
	return &Record{
		Source: opts.Source, License: opts.License,
		Name: f.Name, Hash: f.Hash, Model: opts.Backend.Name(),
		System: SystemPrompt, Input: in.Text, Output: out,
	}, nil
}

// sample escolhe as funções da amostra. A ordem pelo hash do código funciona
// como sorteio que dá sempre o mesmo resultado para o mesmo repositório.
func sample(st *store.Store, opts Options, all map[int64]store.FuncRow) (map[int64]store.FuncRow, error) {
	generated := map[string]bool{}
	isGenerated := func(file string) bool {
		g, ok := generated[file]
		if !ok {
			g = GeneratedFile(filepath.Join(opts.Root, file))
			generated[file] = g
		}
		return g
	}

	var candidates []store.FuncRow
	for _, f := range all {
		if f.LineEnd-f.LineStart+1 >= opts.MinLines && !strings.HasSuffix(f.File, "_test.go") && !isGenerated(f.File) {
			candidates = append(candidates, f)
		}
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].Hash != candidates[j].Hash {
			return candidates[i].Hash < candidates[j].Hash
		}
		return candidates[i].Name < candidates[j].Name
	})
	if len(candidates) > opts.Sample {
		candidates = candidates[:opts.Sample]
	}

	picked := map[int64]store.FuncRow{}
	for _, f := range candidates {
		picked[f.ID] = f
		callees, err := st.Callees(f.ID)
		if err != nil {
			return nil, err
		}
		for _, c := range callees {
			if cf, ok := all[c.ID]; ok && !isGenerated(cf.File) {
				picked[c.ID] = cf
			}
		}
	}
	return picked, nil
}

// GeneratedFile indica arquivos marcados com "Code generated ... DO NOT EDIT.".
func GeneratedFile(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for i := 0; i < 30 && sc.Scan(); i++ {
		line := sc.Text()
		if strings.HasPrefix(line, "// Code generated ") && strings.HasSuffix(line, " DO NOT EDIT.") {
			return true
		}
		if strings.HasPrefix(line, "package ") {
			break
		}
	}
	return false
}
