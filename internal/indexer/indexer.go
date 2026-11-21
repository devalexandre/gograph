// Package indexer monta o grafo de funções de um projeto Go usando o type
// checker oficial, SSA e o call graph VTA.
package indexer

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/callgraph/cha"
	"golang.org/x/tools/go/callgraph/vta"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"

	"github.com/devalexandre/gograph/internal/store"
)

// Options configura a indexação.
type Options struct {
	Dir      string   // raiz do projeto (onde está o go.mod)
	Patterns []string // padrões de pacote; padrão "./..."
	Tests    bool     // incluir arquivos _test.go
	CHAOnly  bool     // pular o VTA e usar só o CHA (mais rápido, menos preciso)

	Env     []string        // ambiente do comando go; nil usa o do processo
	Context context.Context // cancela a carga dos pacotes; nil = sem cancelamento

	// MaxTargets é o máximo de funções que uma única chamada dinâmica pode
	// ter como alvo antes de ser tratada como não resolvida.
	MaxTargets int
}

// DefaultMaxTargets vale quando Options.MaxTargets é zero.
const DefaultMaxTargets = 10

// Timings registra quanto tempo cada etapa levou.
type Timings struct {
	Load, SSA, CallGraph, Extract time.Duration
}

// Result é o grafo pronto para gravar.
type Result struct {
	Module    string
	Functions []store.Function
	Edges     []store.Edge
	Skipped   []SkippedPackage // pacotes do módulo que não compilam
	Timings   Timings
}

// SkippedPackage é um pacote deixado de fora do índice.
type SkippedPackage struct {
	Path   string
	Reason string
}

// Index carrega o projeto e devolve o grafo de funções do módulo principal.
func Index(opts Options) (*Result, error) {
	dir, err := filepath.Abs(opts.Dir)
	if err != nil {
		return nil, err
	}
	if opts.MaxTargets <= 0 {
		opts.MaxTargets = DefaultMaxTargets
	}
	patterns := opts.Patterns
	if len(patterns) == 0 {
		patterns = []string{"./..."}
	}

	res := &Result{}
	start := time.Now()
	cfg := &packages.Config{
		Dir:     dir,
		Tests:   opts.Tests,
		Env:     opts.Env,
		Context: opts.Context,
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
			packages.NeedImports | packages.NeedDeps | packages.NeedTypes |
			packages.NeedSyntax | packages.NeedTypesInfo | packages.NeedTypesSizes |
			packages.NeedModule,
	}
	pkgs, err := packages.Load(cfg, patterns...)
	if err != nil {
		return nil, fmt.Errorf("carregar pacotes: %w", err)
	}
	if len(pkgs) == 0 {
		return nil, fmt.Errorf("nenhum pacote Go encontrado em %s", dir)
	}
	res.Timings.Load = time.Since(start)

	// Pacotes do módulo principal, pelo caminho de import. Pacotes que não
	// compilam (ou que dependem de um que não compila) ficam de fora com um
	// aviso, para um pacote quebrado não impedir a indexação do resto.
	modPkgs := map[string]bool{}
	packages.Visit(pkgs, nil, func(p *packages.Package) {
		if p.Module == nil || !p.Module.Main {
			return
		}
		if res.Module == "" {
			res.Module = p.Module.Path
		}
		if p.Types == nil || p.IllTyped {
			msg := "depende de pacote que não compila"
			if len(p.Errors) > 0 {
				msg = p.Errors[0].Error()
				if n := len(p.Errors) - 1; n > 0 {
					msg += fmt.Sprintf(" (e mais %d)", n)
				}
			}
			res.Skipped = append(res.Skipped, SkippedPackage{Path: p.PkgPath, Reason: msg})
			return
		}
		modPkgs[p.Types.Path()] = true
	})
	sort.Slice(res.Skipped, func(i, j int) bool { return res.Skipped[i].Path < res.Skipped[j].Path })
	if len(modPkgs) == 0 {
		if len(res.Skipped) > 0 {
			return nil, fmt.Errorf("nenhum pacote do módulo compila; primeiro erro: %s", res.Skipped[0].Reason)
		}
		return nil, fmt.Errorf("nenhum pacote do módulo principal em %s", dir)
	}

	start = time.Now()
	prog, _ := ssautil.AllPackages(pkgs, ssa.InstantiateGenerics)
	prog.Build()
	res.Timings.SSA = time.Since(start)

	start = time.Now()
	allFuncs := ssautil.AllFunctions(prog)
	cg := cha.CallGraph(prog)
	if !opts.CHAOnly {
		cg = vta.CallGraph(allFuncs, cg)
	}
	res.Timings.CallGraph = time.Since(start)

	start = time.Now()
	x := &extractor{
		dir:     dir,
		fset:    prog.Fset,
		modPkgs: modPkgs,
		nodes:   map[string]*store.Function{},
		edges:   map[store.Edge]bool{},
		sources: map[string][]byte{},
		seenPos: map[token.Pos]bool{},

		maxTargets: opts.MaxTargets,
	}
	for fn := range allFuncs {
		x.addModuleFunc(fn)
	}
	for fn := range cg.Nodes {
		x.addModuleFunc(fn)
	}
	x.addUnbuiltDecls(pkgs)
	x.addEdges(cg)
	res.Timings.Extract = time.Since(start)

	for _, f := range x.nodes {
		res.Functions = append(res.Functions, *f)
	}
	sort.Slice(res.Functions, func(i, j int) bool { return res.Functions[i].Name < res.Functions[j].Name })
	for e := range x.edges {
		res.Edges = append(res.Edges, e)
	}
	sort.Slice(res.Edges, func(i, j int) bool {
		a, b := res.Edges[i], res.Edges[j]
		if a.Caller != b.Caller {
			return a.Caller < b.Caller
		}
		if a.Callee != b.Callee {
			return a.Callee < b.Callee
		}
		return a.Kind < b.Kind
	})
	return res, nil
}

type extractor struct {
	dir     string
	fset    *token.FileSet
	modPkgs map[string]bool
	nodes   map[string]*store.Function
	edges   map[store.Edge]bool
	sources map[string][]byte
	seenPos map[token.Pos]bool

	maxTargets int
}

// canonical troca uma instância de função genérica pela função genérica de
// origem, que é a que existe no código-fonte.
func canonical(fn *ssa.Function) *ssa.Function {
	if fn == nil {
		return nil
	}
	if o := fn.Origin(); o != nil {
		return o
	}
	return fn
}

// isSynthetic indica funções criadas pelo SSA (wrappers, bounds, thunks e o
// init do pacote), que não existem no código e são atravessadas no grafo.
func isSynthetic(fn *ssa.Function) bool {
	return fn.Synthetic != ""
}

func pkgPathOf(fn *ssa.Function) string {
	for p := fn; p != nil; p = p.Parent() {
		if p.Pkg != nil {
			return p.Pkg.Pkg.Path()
		}
		if obj := p.Object(); obj != nil && obj.Pkg() != nil {
			return obj.Pkg().Path()
		}
	}
	return ""
}

func (x *extractor) inModule(fn *ssa.Function) bool {
	return x.modPkgs[pkgPathOf(fn)]
}

// funcName devolve o nome completo: pacote.Func, pacote.(*Tipo).Metodo ou
// pacote.Tipo.Metodo. Funções anônimas ficam como Pai$1.
func funcName(fn *ssa.Function) string {
	if p := fn.Parent(); p != nil {
		return funcName(p) + strings.TrimPrefix(fn.Name(), p.Name())
	}
	pkg := pkgPathOf(fn)
	if recv := fn.Signature.Recv(); recv != nil {
		t := types.Unalias(recv.Type())
		ptr := false
		if p, ok := t.(*types.Pointer); ok {
			ptr = true
			t = types.Unalias(p.Elem())
		}
		if named, ok := t.(*types.Named); ok {
			if named.Obj().Pkg() != nil {
				pkg = named.Obj().Pkg().Path()
			}
			if ptr {
				return fmt.Sprintf("%s.(*%s).%s", pkg, named.Obj().Name(), fn.Name())
			}
			return fmt.Sprintf("%s.%s.%s", pkg, named.Obj().Name(), fn.Name())
		}
	}
	if pkg == "" {
		return fn.Name()
	}
	return pkg + "." + fn.Name()
}

func (x *extractor) source(file string) []byte {
	if b, ok := x.sources[file]; ok {
		return b
	}
	b, _ := os.ReadFile(file)
	x.sources[file] = b
	return b
}

// describeSyntax preenche arquivo, linhas, assinatura e hash a partir do nó
// da AST (FuncDecl ou FuncLit).
func (x *extractor) describeSyntax(f *store.Function, node ast.Node) {
	var typ *ast.FuncType
	var body *ast.BlockStmt
	start := node.Pos()
	switch n := node.(type) {
	case *ast.FuncDecl:
		typ, body = n.Type, n.Body
	case *ast.FuncLit:
		typ, body = n.Type, n.Body
	default:
		return
	}
	startPos := x.fset.Position(start)
	endPos := x.fset.Position(node.End())
	if !startPos.IsValid() {
		return
	}
	if rel, err := filepath.Rel(x.dir, startPos.Filename); err == nil && !strings.HasPrefix(rel, "..") {
		f.File = rel
	} else {
		f.File = startPos.Filename
	}
	f.LineStart, f.LineEnd = startPos.Line, endPos.Line

	src := x.source(startPos.Filename)
	if endPos.Offset > len(src) || startPos.Offset > endPos.Offset {
		return // arquivo gerado (cgo) ou diferente do que foi compilado
	}
	f.Hash = hashSource(src[startPos.Offset:endPos.Offset])

	sigEnd := x.fset.Position(typ.End()).Offset
	if body != nil {
		sigEnd = x.fset.Position(body.Lbrace).Offset
	}
	if sigEnd > startPos.Offset && sigEnd <= len(src) {
		f.Signature = strings.Join(strings.Fields(string(src[startPos.Offset:sigEnd])), " ")
	}
}

// addModuleFunc registra fn como nó se ela for uma função real do módulo.
func (x *extractor) addModuleFunc(fn *ssa.Function) {
	fn = canonical(fn)
	if fn == nil || isSynthetic(fn) || !x.inModule(fn) {
		return
	}
	name := funcName(fn)
	if _, ok := x.nodes[name]; ok {
		return
	}
	f := &store.Function{Name: name, Package: pkgPathOf(fn)}
	if syn := fn.Syntax(); syn != nil {
		x.describeSyntax(f, syn)
		x.seenPos[syn.Pos()] = true
	}
	x.nodes[name] = f

	// Função anônima fica ligada à função que a contém.
	if p := canonical(fn.Parent()); p != nil && !isSynthetic(p) {
		x.addModuleFunc(p)
		x.edges[store.Edge{Caller: funcName(p), Callee: name, Kind: store.KindClosure}] = true
	}
}

// addUnbuiltDecls inclui declarações que o SSA não constrói (por exemplo
// funções sem uso que ele descarta), para o índice ter todas as funções do
// código.
func (x *extractor) addUnbuiltDecls(pkgs []*packages.Package) {
	packages.Visit(pkgs, nil, func(p *packages.Package) {
		if p.Module == nil || !p.Module.Main || !x.modPkgs[p.Types.Path()] {
			return
		}
		for _, file := range p.Syntax {
			for _, d := range file.Decls {
				decl, ok := d.(*ast.FuncDecl)
				if !ok || decl.Name.Name == "_" || x.seenPos[decl.Pos()] {
					continue
				}
				obj, ok := p.TypesInfo.Defs[decl.Name].(*types.Func)
				if !ok {
					continue
				}
				name := objName(obj)
				if _, exists := x.nodes[name]; exists {
					continue
				}
				f := &store.Function{Name: name, Package: p.Types.Path()}
				x.describeSyntax(f, decl)
				x.nodes[name] = f
			}
		}
	})
}

func objName(obj *types.Func) string {
	pkg := ""
	if obj.Pkg() != nil {
		pkg = obj.Pkg().Path()
	}
	sig, _ := obj.Type().(*types.Signature)
	if sig != nil && sig.Recv() != nil {
		t := types.Unalias(sig.Recv().Type())
		ptr := false
		if p, ok := t.(*types.Pointer); ok {
			ptr = true
			t = types.Unalias(p.Elem())
		}
		if named, ok := t.(*types.Named); ok {
			if ptr {
				return fmt.Sprintf("%s.(*%s).%s", pkg, named.Obj().Name(), obj.Name())
			}
			return fmt.Sprintf("%s.%s.%s", pkg, named.Obj().Name(), obj.Name())
		}
	}
	return pkg + "." + obj.Name()
}

// externalNode registra um nó externo (stdlib ou dependência).
func (x *extractor) externalNode(fn *ssa.Function) string {
	name := funcName(fn)
	if _, ok := x.nodes[name]; !ok {
		sig := types.TypeString(fn.Signature, func(p *types.Package) string { return p.Name() })
		x.nodes[name] = &store.Function{
			Name:      name,
			Package:   pkgPathOf(fn),
			Signature: sig,
			External:  true,
		}
	}
	return name
}

// realTargets segue nós sintéticos até chegar nas funções reais chamadas.
func realTargets(n *callgraph.Node, seen map[*callgraph.Node]bool, out map[*ssa.Function]bool) {
	if n == nil || n.Func == nil || seen[n] {
		return
	}
	seen[n] = true
	fn := canonical(n.Func)
	if !isSynthetic(fn) {
		out[fn] = true
		return
	}
	for _, e := range n.Out {
		realTargets(e.Callee, seen, out)
	}
}

func edgeKind(site ssa.CallInstruction, callee *ssa.Function) string {
	switch site.(type) {
	case *ssa.Go:
		return store.KindGo
	case *ssa.Defer:
		return store.KindDefer
	}
	if site == nil {
		return store.KindStatic
	}
	common := site.Common()
	if common.IsInvoke() {
		return store.KindInterface
	}
	if callee == nil {
		return store.KindInterface
	}
	if callee.Parent() != nil {
		return store.KindClosure
	}
	if common.StaticCallee() != nil {
		return store.KindStatic
	}
	return store.KindDynamic
}

// addEdges percorre o call graph e grava as arestas que saem de funções do
// módulo. Chamadas feitas por código externo (callbacks) não entram.
//
// Um ponto de chamada que resolve para mais de maxTargets funções não foi de
// fato resolvido: é o caso de err.Error() ou w.Write(), em que qualquer tipo
// do programa serve. Nesse caso a chamada via interface vira uma única aresta
// para o método da interface, e a chamada via valor func é descartada.
// Implementações externas de uma interface também viram essa aresta única;
// só implementações do módulo aparecem uma a uma.
func (x *extractor) addEdges(cg *callgraph.Graph) {
	for fn, node := range cg.Nodes {
		caller := canonical(fn)
		if caller == nil || isSynthetic(caller) || !x.inModule(caller) {
			continue
		}
		callerName := funcName(caller)
		if _, ok := x.nodes[callerName]; !ok {
			continue
		}
		bySite := map[ssa.CallInstruction]map[*ssa.Function]bool{}
		for _, e := range node.Out {
			targets := bySite[e.Site]
			if targets == nil {
				targets = map[*ssa.Function]bool{}
				bySite[e.Site] = targets
			}
			realTargets(e.Callee, map[*callgraph.Node]bool{}, targets)
		}
		for site, targets := range bySite {
			invoke := site != nil && site.Common().IsInvoke()
			unresolved := site != nil && len(targets) > x.maxTargets
			viaInterface := false // a chamada precisa da aresta para o método da interface
			if !unresolved {
				for callee := range targets {
					var calleeName string
					switch {
					case x.inModule(callee):
						calleeName = funcName(callee)
						if _, ok := x.nodes[calleeName]; !ok {
							continue
						}
					case invoke:
						// Implementação externa de uma interface: o nome do
						// tipo concreto não ajuda, o método da interface sim.
						viaInterface = true
						continue
					default:
						calleeName = x.externalNode(callee)
					}
					x.edges[store.Edge{Caller: callerName, Callee: calleeName, Kind: edgeKind(site, callee)}] = true
				}
			}
			if invoke && (unresolved || viaInterface) {
				callee := x.interfaceMethodNode(site.Common().Method)
				x.edges[store.Edge{Caller: callerName, Callee: callee, Kind: edgeKind(site, nil)}] = true
			}
		}
	}
}

// interfaceMethodNode registra o método de uma interface (por exemplo
// error.Error) como nó sem corpo, usado quando a chamada não tem um conjunto
// pequeno de implementações possíveis.
func (x *extractor) interfaceMethodNode(m *types.Func) string {
	name := strings.TrimPrefix(objName(m), ".")
	if _, ok := x.nodes[name]; !ok {
		pkg := ""
		if m.Pkg() != nil {
			pkg = m.Pkg().Path()
		}
		x.nodes[name] = &store.Function{
			Name:      name,
			Package:   pkg,
			Signature: types.TypeString(m.Type(), func(p *types.Package) string { return p.Name() }),
			External:  true,
		}
	}
	return name
}
