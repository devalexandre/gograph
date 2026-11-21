package implement

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"golang.org/x/tools/go/packages"
)

// ContextMode define como o contexto é construído para tarefas de escrita.
const (
	CtxReal           = "real"
	CtxOracleDistract = "oracle-distract"
	CtxOracle         = "oracle"
)

var (
	// DefaultCtxMode é o modo padrão usado por BuildTask quando não especificado.
	DefaultCtxMode = CtxReal
	// DefaultBudgetChars é o orçamento padrão de caracteres do contexto em ctx=real e ctx=oracle-distract.
	DefaultBudgetChars = 24000
)

type pkgFile struct {
	pkg  *packages.Package
	file *ast.File
	src  []byte
}

type structFieldEntry struct {
	names []string
	text  string
}

type declEntry struct {
	pkgPath    string
	pkgName    string
	name       string
	exported   bool
	text       string
	file       string
	line       int
	pos        int
	isStruct   bool
	typeHeader string             // "type Foo struct {"
	fields     []structFieldEntry // campos do struct, na ordem de declaração
	funcObj    *types.Func
	recvType   string // nome do tipo do receptor, se método
}

type modCtx struct {
	root   string
	module string
	byFile map[string]pkgFile
	// todos os tipos por pkgPath, na ordem de declaração (arquivo, linha)
	typesByPkg map[string][]declEntry
	typeMap    map[string]map[string]declEntry
	// constantes e variáveis por pkgPath, na ordem de declaração
	valuesByPkg map[string][]declEntry
	valMap      map[string]map[string]declEntry
	// funções e métodos por pkgPath, na ordem de declaração
	funcsByPkg map[string][]declEntry
	funcSigs   map[*types.Func]declEntry
}

type funcContextInfo struct {
	typesDecl   string
	valuesDecl  string
	funcEntries []declEntry
}

var (
	modCtxMu    sync.Mutex
	modCtxCache *modCtx
)

// InvalidateModCtx limpa o cache do módulo (útil em testes que alteram arquivos).
func InvalidateModCtx() {
	modCtxMu.Lock()
	defer modCtxMu.Unlock()
	modCtxCache = nil
}

func defaultGoEnv(root string) []string {
	env := os.Environ()
	if filepath.Base(filepath.Dir(root)) == "src" {
		corpusDir := filepath.Dir(filepath.Dir(root))
		modcache := filepath.Join(corpusDir, "modcache")
		if st, err := os.Stat(modcache); err == nil && st.IsDir() {
			env = append(env,
				"GOMODCACHE="+modcache,
				"GOFLAGS=-mod=mod -modcacherw",
				"GOTOOLCHAIN=local",
				"CGO_ENABLED=0",
				"GOWORK=off",
			)
		}
	}
	return env
}

func loadModCtx(root string) *modCtx {
	absRoot, err := filepath.Abs(root)
	if err == nil {
		root = absRoot
	}
	modCtxMu.Lock()
	defer modCtxMu.Unlock()
	if modCtxCache != nil && modCtxCache.root == root {
		return modCtxCache
	}

	mc := &modCtx{
		root:        root,
		byFile:      map[string]pkgFile{},
		typesByPkg:  map[string][]declEntry{},
		typeMap:     map[string]map[string]declEntry{},
		valuesByPkg: map[string][]declEntry{},
		valMap:      map[string]map[string]declEntry{},
		funcsByPkg:  map[string][]declEntry{},
		funcSigs:    map[*types.Func]declEntry{},
	}

	cfg := &packages.Config{
		Dir: root,
		Env: defaultGoEnv(root),
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
			packages.NeedImports | packages.NeedDeps | packages.NeedTypes |
			packages.NeedSyntax | packages.NeedTypesInfo | packages.NeedModule,
	}
	pkgs, err := packages.Load(cfg, "./...")
	if err != nil {
		modCtxCache = mc
		return mc
	}

	// Ordena pacotes por PkgPath para determinismo
	sort.Slice(pkgs, func(i, j int) bool { return pkgs[i].PkgPath < pkgs[j].PkgPath })

	posSeq := 0
	for _, p := range pkgs {
		if p.Module != nil && mc.module == "" {
			mc.module = p.Module.Path
		}
		if mc.typeMap[p.PkgPath] == nil {
			mc.typeMap[p.PkgPath] = map[string]declEntry{}
		}
		if mc.valMap[p.PkgPath] == nil {
			mc.valMap[p.PkgPath] = map[string]declEntry{}
		}

		// Ordena os arquivos do pacote por nome
		type filePair struct {
			path string
			ast  *ast.File
		}
		var files []filePair
		for i, f := range p.Syntax {
			if i >= len(p.CompiledGoFiles) {
				continue
			}
			fpath := p.CompiledGoFiles[i]
			if abs, err := filepath.Abs(fpath); err == nil {
				fpath = abs
			}
			if strings.HasSuffix(fpath, "_test.go") {
				continue
			}
			files = append(files, filePair{path: fpath, ast: f})
		}
		sort.Slice(files, func(i, j int) bool { return files[i].path < files[j].path })

		for _, fp := range files {
			fpath, f := fp.path, fp.ast
			src, err := os.ReadFile(fpath)
			if err != nil {
				continue
			}
			mc.byFile[fpath] = pkgFile{pkg: p, file: f, src: src}

			nodeText := func(n ast.Node) string {
				start := p.Fset.Position(n.Pos()).Offset
				end := p.Fset.Position(n.End()).Offset
				if start < 0 || end > len(src) || start > end {
					return ""
				}
				return string(src[start:end])
			}

			for _, d := range f.Decls {
				switch decl := d.(type) {
				case *ast.GenDecl:
					for _, spec := range decl.Specs {
						switch s := spec.(type) {
						case *ast.TypeSpec:
							rawTxt := nodeText(s)
							if rawTxt == "" {
								continue
							}
							line := p.Fset.Position(s.Pos()).Line
							de := declEntry{
								pkgPath:  p.PkgPath,
								pkgName:  p.Name,
								name:     s.Name.Name,
								exported: ast.IsExported(s.Name.Name),
								text:     "type " + compactTypeText(rawTxt),
								file:     fpath,
								line:     line,
								pos:      posSeq,
							}
							posSeq++
							if st, ok := s.Type.(*ast.StructType); ok && st.Fields != nil {
								de.isStruct = true
								lbrace := p.Fset.Position(st.Fields.Opening).Offset
								sStart := p.Fset.Position(s.Pos()).Offset
								if lbrace >= sStart && lbrace < len(src) {
									de.typeHeader = "type " + strings.TrimSpace(string(src[sStart:lbrace+1]))
								} else {
									de.typeHeader = "type " + s.Name.Name + " struct {"
								}
								for _, fld := range st.Fields.List {
									ftxt := strings.TrimSpace(compactTypeText(nodeText(fld)))
									if ftxt == "" {
										continue
									}
									var fnames []string
									for _, id := range fld.Names {
										fnames = append(fnames, id.Name)
									}
									if len(fnames) == 0 {
										// Campo embedded
										if id := embeddedIdent(fld.Type); id != "" {
											fnames = append(fnames, id)
										}
									}
									de.fields = append(de.fields, structFieldEntry{
										names: fnames,
										text:  "\t" + ftxt,
									})
								}
							}
							mc.typesByPkg[p.PkgPath] = append(mc.typesByPkg[p.PkgPath], de)
							mc.typeMap[p.PkgPath][de.name] = de

						case *ast.ValueSpec:
							declStr := formatValueSpecOrBlock(p, src, decl, s, nodeText)
							if declStr == "" {
								continue
							}
							line := p.Fset.Position(s.Pos()).Line
							for _, id := range s.Names {
								if id.Name == "_" {
									continue
								}
								ve := declEntry{
									pkgPath:  p.PkgPath,
									pkgName:  p.Name,
									name:     id.Name,
									exported: ast.IsExported(id.Name),
									text:     declStr,
									file:     fpath,
									line:     line,
									pos:      posSeq,
								}
								posSeq++
								if _, exists := mc.valMap[p.PkgPath][id.Name]; !exists {
									mc.valuesByPkg[p.PkgPath] = append(mc.valuesByPkg[p.PkgPath], ve)
								}
								mc.valMap[p.PkgPath][id.Name] = ve
							}
						}
					}
				case *ast.FuncDecl:
					if decl.Name.Name == "init" || decl.Name.Name == "_" {
						continue
					}
					sigTxt := funcHeaderText(p, src, decl)
					if sigTxt == "" {
						continue
					}
					line := p.Fset.Position(decl.Pos()).Line
					rType := recvName(decl)
					exp := ast.IsExported(decl.Name.Name) && (rType == "" || ast.IsExported(rType))
					var fobj *types.Func
					if p.TypesInfo != nil {
						fobj, _ = p.TypesInfo.Defs[decl.Name].(*types.Func)
					}
					fe := declEntry{
						pkgPath:  p.PkgPath,
						pkgName:  p.Name,
						name:     decl.Name.Name,
						exported: exp,
						text:     sigTxt,
						file:     fpath,
						line:     line,
						pos:      posSeq,
						funcObj:  fobj,
						recvType: rType,
					}
					posSeq++
					mc.funcsByPkg[p.PkgPath] = append(mc.funcsByPkg[p.PkgPath], fe)
					if fobj != nil {
						mc.funcSigs[fobj] = fe
					}
				}
			}
		}
	}

	modCtxCache = mc
	return mc
}

func embeddedIdent(expr ast.Expr) string {
	switch x := expr.(type) {
	case *ast.Ident:
		return x.Name
	case *ast.StarExpr:
		return embeddedIdent(x.X)
	case *ast.SelectorExpr:
		return x.Sel.Name
	case *ast.IndexExpr:
		return embeddedIdent(x.X)
	case *ast.IndexListExpr:
		return embeddedIdent(x.X)
	}
	return ""
}

func funcHeaderText(p *packages.Package, src []byte, decl *ast.FuncDecl) string {
	start := p.Fset.Position(decl.Pos()).Offset
	var end int
	if decl.Body != nil {
		end = p.Fset.Position(decl.Body.Lbrace).Offset
	} else {
		end = p.Fset.Position(decl.End()).Offset
	}
	if start < 0 || end > len(src) || start >= end {
		return ""
	}
	return strings.TrimSpace(string(src[start:end]))
}

// formatValueSpecOrBlock formata cada ValueSpec de um bloco const (...) ou var (...)
// individualmente (uma declaração por spec), mantendo todos os identificadores do spec,
// cortando valores longos (>1 linha ou >120 chars -> "= ...") e mantendo o comentário
// do bloco/spec compactado em 1 linha. Assim todos os identificadores do bloco entram
// na lista do pacote sem que um único ValueSpec replique centenas de linhas do bloco inteiro.
func formatValueSpecOrBlock(p *packages.Package, src []byte, decl *ast.GenDecl, spec *ast.ValueSpec, nodeText func(ast.Node) string) string {
	srcLines := strings.Split(string(src), "\n")
	extractDocBefore := func(line1Idx int) string {
		var comments []string
		for idx := line1Idx - 2; idx >= 0; idx-- {
			t := strings.TrimSpace(srcLines[idx])
			if strings.HasPrefix(t, "//") {
				c := strings.TrimSpace(strings.TrimPrefix(t, "//"))
				if c != "" {
					comments = append([]string{c}, comments...)
				}
			} else {
				break
			}
		}
		if len(comments) == 0 {
			return ""
		}
		joined := strings.Join(comments, " ")
		if len(joined) > 100 {
			joined = joined[:97] + "..."
		}
		return "// " + joined
	}

	var names []string
	for _, id := range spec.Names {
		if id.Name != "_" {
			names = append(names, id.Name)
		}
	}
	if len(names) == 0 {
		return ""
	}
	tok := decl.Tok.String()
	var b strings.Builder
	b.WriteString(tok + " " + strings.Join(names, ", "))
	if spec.Type != nil {
		tTxt := strings.TrimSpace(nodeText(spec.Type))
		if strings.Contains(tTxt, "\n") || len(tTxt) > 120 {
			tTxt = "..."
		}
		if tTxt != "" {
			b.WriteString(" " + tTxt)
		}
	}
	if len(spec.Values) > 0 {
		var vals []string
		long := false
		for _, v := range spec.Values {
			vTxt := strings.TrimSpace(nodeText(v))
			if strings.Contains(vTxt, "\n") || len(vTxt) > 120 {
				long = true
				break
			}
			vals = append(vals, vTxt)
		}
		valJoined := strings.Join(vals, ", ")
		if long || len(valJoined) > 120 {
			b.WriteString(" = ...")
		} else if valJoined != "" {
			b.WriteString(" = " + alignRE.ReplaceAllString(valJoined, " "))
		}
	}
	sLine := b.String()
	// Mantém o comentário de bloco (compactado em 1 linha) apenas no início do bloco const/var
	isFirstInBlock := len(decl.Specs) > 0 && decl.Specs[0] == spec
	if isFirstInBlock {
		if doc := extractDocBefore(p.Fset.Position(decl.Pos()).Line); doc != "" {
			return doc + "\n" + sLine
		}
	}
	return sLine
}

// compactTypeText remove linhas vazias e linhas de comentário puro de dentro
// de structs/interfaces para que tipos grandes caibam no orçamento.
func compactTypeText(s string) string {
	if s == "" {
		return ""
	}
	lines := strings.Split(s, "\n")
	var out []string
	for _, l := range lines {
		trimmed := strings.TrimSpace(l)
		if trimmed == "" || strings.HasPrefix(trimmed, "//") {
			continue
		}
		out = append(out, l)
	}
	return strings.Join(out, "\n")
}

// formatStruct formata um struct respeitando maxLines:
// - Se o struct inteiro cabe em maxLines, devolve inteiro.
// - Se não cabe: inclui todos os campos cujo nome esteja em reqFields, e
//   completa com os primeiros campos em ordem de declaração até atingir o
//   limite de linhas; os campos restantes viram "// ... mais N campos".
func formatStruct(de declEntry, maxLines int, reqFields map[string]bool) string {
	if !de.isStruct || len(de.fields) == 0 {
		return clip(de.text, maxLines)
	}
	totalFields := len(de.fields)
	// Verifica se cabe inteiro
	if strings.Count(de.text, "\n")+1 <= maxLines {
		return de.text
	}

	budgetLines := maxLines - 3 // header, "// ... mais N campos", "}"
	if budgetLines < 1 {
		budgetLines = 1
	}

	keep := make([]bool, totalFields)
	usedLines := 0
	// 1. Marca campos obrigatórios (em ctx=oracle / oracle-distract)
	if len(reqFields) > 0 {
		for i, f := range de.fields {
			for _, fn := range f.names {
				if reqFields[fn] {
					keep[i] = true
					usedLines += strings.Count(f.text, "\n") + 1
					break
				}
			}
		}
	}
	// 2. Completa com os primeiros campos na ordem de declaração até o orçamento
	for i, f := range de.fields {
		if keep[i] {
			continue
		}
		flines := strings.Count(f.text, "\n") + 1
		if usedLines+flines <= budgetLines {
			keep[i] = true
			usedLines += flines
		}
	}

	var out []string
	out = append(out, de.typeHeader)
	keptCount := 0
	for i, f := range de.fields {
		if keep[i] {
			out = append(out, f.text)
			keptCount++
		}
	}
	omitted := totalFields - keptCount
	if omitted > 0 {
		out = append(out, fmt.Sprintf("\t// ... mais %d campos", omitted))
	}
	out = append(out, "}")
	return strings.Join(out, "\n")
}

func formatStructToBudget(de declEntry, maxLines int, maxChars int, reqFields map[string]bool, prefix string) (string, bool) {
	if maxChars <= 0 {
		return "", false
	}
	wrap := func(s string) string {
		if prefix != "" {
			return prefix + s
		}
		return s
	}
	txt := wrap(formatStruct(de, maxLines, reqFields))
	if len(txt) <= maxChars {
		return txt, true
	}
	if de.isStruct && len(de.fields) > 0 {
		for ml := maxLines - 5; ml >= 4; ml -= 5 {
			t := wrap(formatStruct(de, ml, reqFields))
			if len(t) <= maxChars {
				return t, true
			}
		}
		stub := wrap(strings.TrimSpace(de.typeHeader) + " ... }")
		if len(stub) <= maxChars {
			return stub, true
		}
		return "", false
	}
	lines := strings.Split(txt, "\n")
	for n := len(lines) - 1; n >= 1; n-- {
		t := strings.Join(lines[:n], "\n")
		if de.typeHeader != "" && strings.Contains(de.typeHeader, "{") {
			t += "\n\t// ...\n}"
		}
		if len(t) <= maxChars {
			return t, true
		}
	}
	if de.typeHeader != "" && strings.Contains(de.typeHeader, "{") {
		stub := wrap(strings.TrimSpace(de.typeHeader) + " ... }")
		if len(stub) <= maxChars {
			return stub, true
		}
	}
	return "", false
}

func (mc *modCtx) isModulePkg(pkgPath string) bool {
	if pkgPath == "" {
		return false
	}
	if _, ok := mc.typeMap[pkgPath]; ok {
		return true
	}
	return mc.module != "" && (pkgPath == mc.module || strings.HasPrefix(pkgPath, mc.module+"/"))
}

// extractRealContext constrói o contexto em modo ctx=real:
// NADA depende do corpo (target.Body) da função-alvo.
func (mc *modCtx) extractRealContext(absFile string, lineStart int, activeImports map[string]bool, budgetChars int) funcContextInfo {
	if abs, err := filepath.Abs(absFile); err == nil {
		absFile = abs
	}
	pf, ok := mc.byFile[absFile]
	if !ok || pf.pkg == nil || pf.pkg.TypesInfo == nil {
		return funcContextInfo{}
	}
	ownPkg := pf.pkg
	info := ownPkg.TypesInfo

	var target *ast.FuncDecl
	for _, d := range pf.file.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok {
			startLine := ownPkg.Fset.Position(fd.Pos()).Line
			if startLine == lineStart {
				target = fd
				break
			}
		}
	}
	if target == nil {
		return funcContextInfo{}
	}

	selfObj := info.Defs[target.Name]
	targetFuncName := target.Name.Name
	targetRecv := recvName(target)

	if budgetChars <= 0 {
		budgetChars = DefaultBudgetChars
	}
	usedChars := 0
	canFit := func(s string) bool {
		return usedChars+len(s)+2 <= budgetChars
	}

	type itemKey struct {
		kind    string // "type", "val", "func"
		pkgPath string
		name    string
		pos     int
	}
	seenTypes := map[string]bool{}
	seenVals := map[string]bool{}
	seenFuncs := map[string]bool{}

	var pickedTypes []declEntry
	var pickedVals []declEntry
	var pickedFuncs []declEntry

	addTypeEntry := func(de declEntry, maxL int, shrink bool) {
		key := de.pkgPath + "." + de.name
		if seenTypes[key] {
			return
		}
		pkgPrefix := ""
		if de.pkgPath != ownPkg.PkgPath && de.pkgName != "" {
			pkgPrefix = fmt.Sprintf("// pacote %s\n", de.pkgName)
		}
		rem := budgetChars - usedChars - 2
		var txt string
		var ok bool
		if shrink {
			txt, ok = formatStructToBudget(de, maxL, rem, nil, pkgPrefix)
		} else {
			txt = pkgPrefix + formatStruct(de, maxL, nil)
			ok = len(txt) <= rem
		}
		if !ok {
			return
		}
		seenTypes[key] = true
		usedChars += len(txt) + 2
		deCopy := de
		deCopy.text = txt
		pickedTypes = append(pickedTypes, deCopy)
	}

	addValEntry := func(ve declEntry) {
		key := ve.pkgPath + "." + ve.name
		if seenVals[key] {
			return
		}
		txt := ve.text
		if ve.pkgPath != ownPkg.PkgPath && ve.pkgName != "" {
			txt = fmt.Sprintf("// pacote %s: %s", ve.pkgName, txt)
		}
		if !canFit(txt) {
			return
		}
		seenVals[key] = true
		usedChars += len(txt) + 1
		veCopy := ve
		veCopy.text = txt
		pickedVals = append(pickedVals, veCopy)
	}

	addFuncEntry := func(fe declEntry) {
		if fe.file == absFile && fe.name == targetFuncName && fe.recvType == targetRecv {
			return // exclui a própria função-alvo
		}
		key := fe.pkgPath + "." + fe.recvType + "." + fe.name
		if seenFuncs[key] {
			return
		}
		txt := fe.text
		if fe.pkgPath != ownPkg.PkgPath && fe.pkgName != "" {
			txt = fmt.Sprintf("// pacote %s: %s", fe.pkgName, txt)
		}
		if !canFit(txt) {
			return
		}
		seenFuncs[key] = true
		usedChars += len(txt) + 3
		feCopy := fe
		feCopy.text = txt
		pickedFuncs = append(pickedFuncs, feCopy)
	}

	// PRIORIDADE 1: Tipos da assinatura em largura (BFS), nível por nível
	type bfsNode struct {
		t     types.Type
		depth int
	}
	var queue []bfsNode
	seenBFS := map[string]bool{}
	var enqueueNamed func(t types.Type, depth int)
	enqueueNamed = func(t types.Type, depth int) {
		if t == nil || depth > 2 {
			return
		}
		switch tt := t.(type) {
		case *types.Pointer:
			enqueueNamed(tt.Elem(), depth)
		case *types.Slice:
			enqueueNamed(tt.Elem(), depth)
		case *types.Array:
			enqueueNamed(tt.Elem(), depth)
		case *types.Map:
			enqueueNamed(tt.Key(), depth)
			enqueueNamed(tt.Elem(), depth)
		case *types.Chan:
			enqueueNamed(tt.Elem(), depth)
		case *types.Named:
			obj := tt.Obj()
			if obj == nil || obj.Pkg() == nil {
				return
			}
			pkgPath := obj.Pkg().Path()
			if !mc.isModulePkg(pkgPath) {
				return
			}
			k := pkgPath + "." + obj.Name()
			if seenBFS[k] {
				return
			}
			seenBFS[k] = true
			queue = append(queue, bfsNode{t: tt, depth: depth})
		}
	}

	if selfFunc, ok := selfObj.(*types.Func); ok && selfFunc != nil {
		if sig, ok := selfFunc.Type().(*types.Signature); ok {
			if sig.Recv() != nil {
				enqueueNamed(sig.Recv().Type(), 0)
			}
			if sig.Params() != nil {
				for i := 0; i < sig.Params().Len(); i++ {
					enqueueNamed(sig.Params().At(i).Type(), 0)
				}
			}
			if sig.Results() != nil {
				for i := 0; i < sig.Results().Len(); i++ {
					enqueueNamed(sig.Results().At(i).Type(), 0)
				}
			}
		}
	}

	for head := 0; head < len(queue); head++ {
		cur := queue[head]
		nt := cur.t.(*types.Named)
		obj := nt.Obj()
		pkgPath := obj.Pkg().Path()
		if pm := mc.typeMap[pkgPath]; pm != nil {
			if de, ok := pm[obj.Name()]; ok {
				addTypeEntry(de, maxTypeLines, true)
			}
		}
		if cur.depth < 2 {
			switch u := nt.Underlying().(type) {
			case *types.Struct:
				for i := 0; i < u.NumFields(); i++ {
					enqueueNamed(u.Field(i).Type(), cur.depth+1)
				}
			case *types.Interface:
				for i := 0; i < u.NumEmbeddeds(); i++ {
					enqueueNamed(u.EmbeddedType(i), cur.depth+1)
				}
			}
		}
	}

	// PRIORIDADE 2: Mesmo arquivo (tipos, constantes/variáveis, funções/métodos)
	for _, de := range mc.typesByPkg[ownPkg.PkgPath] {
		if de.file == absFile {
			addTypeEntry(de, maxTypeLines, false)
		}
	}
	for _, ve := range mc.valuesByPkg[ownPkg.PkgPath] {
		if ve.file == absFile {
			addValEntry(ve)
		}
	}
	for _, fe := range mc.funcsByPkg[ownPkg.PkgPath] {
		if fe.file == absFile {
			addFuncEntry(fe)
		}
	}

	// PRIORIDADE 3: Mesmo pacote (demais arquivos do pacote: tipos, constantes/variáveis, métodos do receptor primeiro e demais funções)
	for _, de := range mc.typesByPkg[ownPkg.PkgPath] {
		if de.file != absFile {
			addTypeEntry(de, 40, false)
		}
	}
	for _, ve := range mc.valuesByPkg[ownPkg.PkgPath] {
		if ve.file != absFile {
			addValEntry(ve)
		}
	}
	if targetRecv != "" {
		for _, fe := range mc.funcsByPkg[ownPkg.PkgPath] {
			if fe.file != absFile && fe.recvType == targetRecv {
				addFuncEntry(fe)
			}
		}
	}
	for _, fe := range mc.funcsByPkg[ownPkg.PkgPath] {
		if fe.file != absFile {
			addFuncEntry(fe)
		}
	}

	// PRIORIDADE 4: API exportada dos pacotes do módulo importados pelo arquivo (sem a função-alvo)
	var impPkgs []string
	for impPath := range activeImports {
		if mc.isModulePkg(impPath) && impPath != ownPkg.PkgPath {
			impPkgs = append(impPkgs, impPath)
		}
	}
	sort.Strings(impPkgs)
	for _, impPath := range impPkgs {
		for _, de := range mc.typesByPkg[impPath] {
			if de.exported {
				addTypeEntry(de, 30, false)
			}
		}
		for _, ve := range mc.valuesByPkg[impPath] {
			if ve.exported {
				addValEntry(ve)
			}
		}
		for _, fe := range mc.funcsByPkg[impPath] {
			if fe.exported {
				addFuncEntry(fe)
			}
		}
	}

	// Ordena tudo estritamente por ordem de declaração (pos: arquivo e linha)
	sort.Slice(pickedTypes, func(i, j int) bool { return pickedTypes[i].pos < pickedTypes[j].pos })
	sort.Slice(pickedVals, func(i, j int) bool { return pickedVals[i].pos < pickedVals[j].pos })
	// pickedFuncs mantém a ordem de prioridade (mesmo arquivo -> receptor -> mesmo pacote -> outros pacotes);
	// a ordenação final por declaração (pos) é feita em BuildTaskWithMode após o corte pelo orçamento total.

	var typeTxts []string
	seenT := map[string]bool{}
	for _, de := range pickedTypes {
		if !seenT[de.text] {
			seenT[de.text] = true
			typeTxts = append(typeTxts, de.text)
		}
	}
	var valTxts []string
	seenV := map[string]bool{}
	for _, ve := range pickedVals {
		if !seenV[ve.text] {
			seenV[ve.text] = true
			valTxts = append(valTxts, ve.text)
		}
	}
	var funcEntries []declEntry
	seenF := map[string]bool{}
	for _, fe := range pickedFuncs {
		if fe.text != "" && !seenF[fe.text] {
			seenF[fe.text] = true
			funcEntries = append(funcEntries, fe)
		}
	}

	return funcContextInfo{
		typesDecl:   strings.Join(typeTxts, "\n\n"),
		valuesDecl:  strings.Join(valTxts, "\n"),
		funcEntries: funcEntries,
	}
}

// extractOracleContext constrói o contexto para ctx=oracle (withDistractors=false)
// ou ctx=oracle-distract (withDistractors=true).
// No modo oracle-distract:
// - inclui obrigatoriamente todos os símbolos usados pelo corpo (missing == 0),
//   mesmo que estejam fora da prioridade ou do orçamento;
// - preenche com distratores ATÉ o orçamento (budgetChars), seguindo a mesma
//   prioridade do ctx=real (tipos da assinatura > mesmo arquivo > mesmo pacote >
//   outros pacotes do módulo), exigindo que os distratores sejam >= ao número de
//   símbolos usados;
// - ordena tudo estritamente por ordem de declaração (arquivo e linha), misturando
//   usados e distratores sem nenhuma marcação que os diferencie.
func (mc *modCtx) extractOracleContext(absFile string, lineStart int, withDistractors bool, budgetChars int) funcContextInfo {
	if abs, err := filepath.Abs(absFile); err == nil {
		absFile = abs
	}
	pf, ok := mc.byFile[absFile]
	if !ok || pf.pkg == nil || pf.pkg.TypesInfo == nil {
		return funcContextInfo{}
	}
	ownPkg := pf.pkg
	info := ownPkg.TypesInfo

	var target *ast.FuncDecl
	for _, d := range pf.file.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok {
			startLine := ownPkg.Fset.Position(fd.Pos()).Line
			endLine := ownPkg.Fset.Position(fd.End()).Line
			if startLine == lineStart || (startLine <= lineStart && lineStart <= endLine) {
				target = fd
				if startLine == lineStart {
					break
				}
			}
		}
	}
	if target == nil {
		return funcContextInfo{}
	}

	selfObj := info.Defs[target.Name]
	targetFuncName := target.Name.Name
	targetRecv := recvName(target)

	if budgetChars <= 0 {
		budgetChars = DefaultBudgetChars
	}

	type key struct {
		pkgPath string
		name    string
	}
	usedTypes := map[key]bool{}
	sigTypes := map[key]bool{}
	usedFields := map[key]map[string]bool{} // struct key -> set of field names used
	usedVals := map[key]bool{}
	usedFuncs := map[string]declEntry{} // key: pkgPath.recv.name

	markField := func(tk key, fieldName string) {
		if usedFields[tk] == nil {
			usedFields[tk] = map[string]bool{}
		}
		usedFields[tk][fieldName] = true
	}

	var addNamedType func(t types.Type, depth int, isSig bool)
	addNamedType = func(t types.Type, depth int, isSig bool) {
		if t == nil || depth > 2 {
			return
		}
		switch tt := t.(type) {
		case *types.Pointer:
			addNamedType(tt.Elem(), depth, isSig)
		case *types.Slice:
			addNamedType(tt.Elem(), depth, isSig)
		case *types.Array:
			addNamedType(tt.Elem(), depth, isSig)
		case *types.Map:
			addNamedType(tt.Key(), depth, isSig)
			addNamedType(tt.Elem(), depth, isSig)
		case *types.Chan:
			addNamedType(tt.Elem(), depth, isSig)
		case *types.Named:
			obj := tt.Obj()
			if obj == nil || obj.Pkg() == nil {
				return
			}
			pkgPath := obj.Pkg().Path()
			if !mc.isModulePkg(pkgPath) {
				return
			}
			k := key{pkgPath: pkgPath, name: obj.Name()}
			if isSig {
				sigTypes[k] = true
			}
			if !usedTypes[k] {
				usedTypes[k] = true
				if depth < 2 {
					switch u := tt.Underlying().(type) {
					case *types.Struct:
						for i := 0; i < u.NumFields(); i++ {
							addNamedType(u.Field(i).Type(), depth+1, isSig)
						}
					case *types.Interface:
						for i := 0; i < u.NumEmbeddeds(); i++ {
							addNamedType(u.EmbeddedType(i), depth+1, isSig)
						}
					}
				}
			}
		}
	}

	if selfFunc, ok := selfObj.(*types.Func); ok && selfFunc != nil {
		if sig, ok := selfFunc.Type().(*types.Signature); ok {
			if sig.Recv() != nil {
				addNamedType(sig.Recv().Type(), 0, true)
			}
			if sig.Params() != nil {
				for i := 0; i < sig.Params().Len(); i++ {
					addNamedType(sig.Params().At(i).Type(), 0, true)
				}
			}
			if sig.Results() != nil {
				for i := 0; i < sig.Results().Len(); i++ {
					addNamedType(sig.Results().At(i).Type(), 0, true)
				}
			}
		}
	}

	qual := func(p *types.Package) string {
		if p == nil || p.Path() == ownPkg.PkgPath {
			return ""
		}
		return p.Name()
	}

	formatTypesFunc := func(fn *types.Func) string {
		sig, ok := fn.Type().(*types.Signature)
		if !ok {
			return ""
		}
		var b bytes.Buffer
		b.WriteString("func ")
		if recv := sig.Recv(); recv != nil {
			b.WriteString("(")
			if recv.Name() != "" {
				b.WriteString(recv.Name() + " ")
			}
			b.WriteString(types.TypeString(recv.Type(), qual))
			b.WriteString(") ")
		} else if fn.Pkg() != nil && fn.Pkg().Path() != ownPkg.PkgPath {
			b.WriteString(fn.Pkg().Name() + ".")
		}
		b.WriteString(fn.Name())
		types.WriteSignature(&b, sig, qual)
		return b.String()
	}

	ast.Inspect(target, func(n ast.Node) bool {
		if n == nil {
			return true
		}
		if expr, ok := n.(ast.Expr); ok {
			if tv, ok := info.Types[expr]; ok && tv.Type != nil {
				addNamedType(tv.Type, 1, false)
			}
		}
		if sel, ok := n.(*ast.SelectorExpr); ok {
			if selection, ok := info.Selections[sel]; ok && selection != nil {
				addNamedType(selection.Recv(), 0, false)
			}
		}
		id, ok := n.(*ast.Ident)
		if !ok {
			return true
		}
		obj := info.Uses[id]
		if obj == nil || obj == selfObj || obj.Pkg() == nil {
			return true
		}
		if obj.Pos() >= target.Pos() && obj.Pos() <= target.End() {
			return true
		}
		if _, isPkg := obj.(*types.PkgName); isPkg {
			return true
		}

		pkgPath := obj.Pkg().Path()
		inMod := mc.isModulePkg(pkgPath)
		if !inMod {
			return true
		}

		switch o := obj.(type) {
		case *types.TypeName:
			addNamedType(o.Type(), 0, false)
		case *types.Const:
			usedVals[key{pkgPath: pkgPath, name: o.Name()}] = true
			addNamedType(o.Type(), 1, false)
		case *types.Var:
			if o.IsField() {
				mc.addFieldOwner(pkgPath, o, func(p, name string) {
					tk := key{pkgPath: p, name: name}
					usedTypes[tk] = true
					markField(tk, o.Name())
				})
				addNamedType(o.Type(), 1, false)
			} else {
				usedVals[key{pkgPath: pkgPath, name: o.Name()}] = true
				addNamedType(o.Type(), 1, false)
			}
		case *types.Func:
			fe, hasFE := mc.funcSigs[o]
			if !hasFE {
				fe = declEntry{
					pkgPath: pkgPath,
					pkgName: obj.Pkg().Name(),
					name:    o.Name(),
					text:    formatTypesFunc(o),
					pos:     999999,
				}
			}
			fkey := fe.pkgPath + "." + fe.recvType + "." + fe.name
			usedFuncs[fkey] = fe
			if sig, ok := o.Type().(*types.Signature); ok && sig.Recv() != nil {
				addNamedType(sig.Recv().Type(), 0, false)
			}
		}
		return true
	})

	usedChars := 0
	pickedTypes := map[key]declEntry{}
	pickedVals := map[key]declEntry{}
	pickedFuncs := map[string]declEntry{}
	var funcPriorityOrder []declEntry

	addType := func(tk key, de declEntry, maxL int, shrink bool, allowOver bool) bool {
		if pickedTypes[tk].isValid() {
			return false
		}
		pkgPrefix := ""
		if de.pkgPath != ownPkg.PkgPath && de.pkgName != "" {
			pkgPrefix = fmt.Sprintf("// pacote %s\n", de.pkgName)
		}
		rem := budgetChars - usedChars - 2
		var txt string
		var ok bool
		if allowOver {
			txt = pkgPrefix + formatStruct(de, maxL, usedFields[tk])
			ok = true
		} else if shrink {
			txt, ok = formatStructToBudget(de, maxL, rem, usedFields[tk], pkgPrefix)
		} else {
			txt = pkgPrefix + formatStruct(de, maxL, usedFields[tk])
			ok = len(txt) <= rem
		}
		if !ok {
			return false
		}
		deCopy := de
		deCopy.text = txt
		pickedTypes[tk] = deCopy
		usedChars += len(txt) + 2
		return true
	}

	addVal := func(vk key, ve declEntry, allowOver bool) bool {
		if pickedVals[vk].isValid() {
			return false
		}
		txt := ve.text
		if ve.pkgPath != ownPkg.PkgPath && ve.pkgName != "" {
			txt = fmt.Sprintf("// pacote %s: %s", ve.pkgName, txt)
		}
		if !allowOver && usedChars+len(txt)+1 > budgetChars {
			return false
		}
		veCopy := ve
		veCopy.text = txt
		pickedVals[vk] = veCopy
		usedChars += len(txt) + 1
		return true
	}

	addFunc := func(fe declEntry, mandatory bool, allowOver bool) bool {
		if fe.file == absFile && fe.name == targetFuncName && fe.recvType == targetRecv {
			return false
		}
		fk := fe.pkgPath + "." + fe.recvType + "." + fe.name
		if _, exists := pickedFuncs[fk]; exists {
			return false
		}
		txt := fe.text
		if fe.pkgPath != ownPkg.PkgPath && fe.pkgName != "" {
			txt = fmt.Sprintf("// pacote %s: %s", fe.pkgName, txt)
		}
		if !allowOver && usedChars+len(txt)+3 > budgetChars {
			return false
		}
		feCopy := fe
		feCopy.text = txt
		feCopy.exported = mandatory
		pickedFuncs[fk] = feCopy
		funcPriorityOrder = append(funcPriorityOrder, feCopy)
		usedChars += len(txt) + 3
		return true
	}

	allowOver := !withDistractors // em ctx=oracle-distract o orçamento é teto duro

	// PASSO 1: Símbolos USADOS pelo corpo têm prioridade máxima
	for tk := range usedTypes {
		if pm := mc.typeMap[tk.pkgPath]; pm != nil {
			if de, ok := pm[tk.name]; ok {
				addType(tk, de, maxTypeLines, true, allowOver)
			}
		}
	}
	for vk := range usedVals {
		if pm := mc.valMap[vk.pkgPath]; pm != nil {
			if ve, ok := pm[vk.name]; ok {
				addVal(vk, ve, allowOver)
			}
		}
	}
	for _, fe := range usedFuncs {
		addFunc(fe, true, allowOver)
	}

	// PASSO 2: Se withDistractors (ctx=oracle-distract), preenche com distratores ATÉ o orçamento (budgetChars)
	if withDistractors {
		tryAddTypeDistractor := func(de declEntry, maxL int) {
			tk := key{pkgPath: de.pkgPath, name: de.name}
			addType(tk, de, maxL, false, false)
		}
		tryAddValDistractor := func(ve declEntry) {
			vk := key{pkgPath: ve.pkgPath, name: ve.name}
			addVal(vk, ve, false)
		}
		tryAddFuncDistractor := func(fe declEntry) {
			addFunc(fe, false, false)
		}

		// Prioridade 1: Tipos da assinatura (já entraram em usedTypes, mas garante subtipos)
		for tk := range sigTypes {
			if pm := mc.typeMap[tk.pkgPath]; pm != nil {
				if de, ok := pm[tk.name]; ok {
					tryAddTypeDistractor(de, maxTypeLines)
				}
			}
		}

		// Prioridade 2: Mesmo arquivo (tipos, constantes/variáveis, funções/métodos)
		for _, de := range mc.typesByPkg[ownPkg.PkgPath] {
			if de.file == absFile {
				tryAddTypeDistractor(de, maxTypeLines)
			}
		}
		for _, ve := range mc.valuesByPkg[ownPkg.PkgPath] {
			if ve.file == absFile {
				tryAddValDistractor(ve)
			}
		}
		for _, fe := range mc.funcsByPkg[ownPkg.PkgPath] {
			if fe.file == absFile {
				tryAddFuncDistractor(fe)
			}
		}

		// Prioridade 3: Mesmo pacote (demais arquivos do pacote: tipos, constantes/variáveis, métodos do receptor primeiro e demais funções)
		for _, de := range mc.typesByPkg[ownPkg.PkgPath] {
			if de.file != absFile {
				tryAddTypeDistractor(de, 40)
			}
		}
		for _, ve := range mc.valuesByPkg[ownPkg.PkgPath] {
			if ve.file != absFile {
				tryAddValDistractor(ve)
			}
		}
		if targetRecv != "" {
			for _, fe := range mc.funcsByPkg[ownPkg.PkgPath] {
				if fe.file != absFile && fe.recvType == targetRecv {
					tryAddFuncDistractor(fe)
				}
			}
		}
		for _, fe := range mc.funcsByPkg[ownPkg.PkgPath] {
			if fe.file != absFile {
				tryAddFuncDistractor(fe)
			}
		}

		// Prioridade 4: Outros pacotes do módulo importados pelo arquivo
		var impPkgPaths []string
		for _, imp := range pf.file.Imports {
			p := strings.Trim(imp.Path.Value, `"`)
			if mc.isModulePkg(p) && p != ownPkg.PkgPath {
				impPkgPaths = append(impPkgPaths, p)
			}
		}
		sort.Strings(impPkgPaths)
		for _, impP := range impPkgPaths {
			for _, de := range mc.typesByPkg[impP] {
				if de.exported {
					tryAddTypeDistractor(de, 30)
				}
			}
			for _, ve := range mc.valuesByPkg[impP] {
				if ve.exported {
					tryAddValDistractor(ve)
				}
			}
			for _, fe := range mc.funcsByPkg[impP] {
				if fe.exported {
					tryAddFuncDistractor(fe)
				}
			}
		}
	}

	// Ordena tipos e valores estritamente por ordem de declaração (pos: arquivo e linha)
	var typeList []declEntry
	for _, de := range pickedTypes {
		typeList = append(typeList, de)
	}
	sort.Slice(typeList, func(i, j int) bool { return typeList[i].pos < typeList[j].pos })

	var valList []declEntry
	for _, ve := range pickedVals {
		valList = append(valList, ve)
	}
	sort.Slice(valList, func(i, j int) bool { return valList[i].pos < valList[j].pos })

	var typeTxts []string
	seenT := map[string]bool{}
	for _, de := range typeList {
		if de.text != "" && !seenT[de.text] {
			seenT[de.text] = true
			typeTxts = append(typeTxts, de.text)
		}
	}
	var valTxts []string
	seenV := map[string]bool{}
	for _, ve := range valList {
		if ve.text != "" && !seenV[ve.text] {
			seenV[ve.text] = true
			valTxts = append(valTxts, ve.text)
		}
	}

	// funcPriorityOrder mantém os usados primeiro e depois distratores em ordem de prioridade;
	// BuildTaskWithMode corta distratores que passarem do budgetChars após enriquecer com descrições
	// e em seguida ordena todas as funções mantidas estritamente por pos (arquivo e linha).
	var funcEntries []declEntry
	seenF := map[string]bool{}
	for _, fe := range funcPriorityOrder {
		if fe.text != "" && !seenF[fe.text] {
			seenF[fe.text] = true
			funcEntries = append(funcEntries, fe)
		}
	}

	return funcContextInfo{
		typesDecl:   strings.Join(typeTxts, "\n\n"),
		valuesDecl:  strings.Join(valTxts, "\n"),
		funcEntries: funcEntries,
	}
}

func (de declEntry) isValid() bool {
	return de.name != ""
}

func (mc *modCtx) addFieldOwner(pkgPath string, field *types.Var, add func(pkgPath, name string)) {
	for _, pf := range mc.byFile {
		if pf.pkg.PkgPath != pkgPath || pf.pkg.Types == nil {
			continue
		}
		scope := pf.pkg.Types.Scope()
		for _, name := range scope.Names() {
			tn, ok := scope.Lookup(name).(*types.TypeName)
			if !ok {
				continue
			}
			st, ok := tn.Type().Underlying().(*types.Struct)
			if !ok {
				continue
			}
			for i := 0; i < st.NumFields(); i++ {
				if st.Field(i) == field || st.Field(i).Pos() == field.Pos() {
					add(pkgPath, tn.Name())
					return
				}
			}
		}
		break
	}
}
