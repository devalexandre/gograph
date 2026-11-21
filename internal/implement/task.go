// Package implement mede se um modelo consegue escrever uma função Go a partir
// da assinatura e do contexto que o grafo fornece.
package implement

import (
	"bytes"
	"database/sql"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/imports"

	"github.com/devalexandre/gograph/internal/describe"
	"github.com/devalexandre/gograph/internal/store"
)

// SystemPrompt é a instrução da tarefa de escrita. O formato da entrada
// (BuildTask) e este texto precisam ser os mesmos no treino e na inferência.
const SystemPrompt = "Você escreve funções Go. Responda só com o código completo da função pedida, sem explicação."

const (
	maxAvailable = 16 // funções listadas como disponíveis
	maxTypeLines = 80 // linhas máximas de cada declaração de tipo
	maxValues    = 24 // constantes e variáveis listadas
	maxValueLines = 6 // linhas de cada uma
)

// Task é uma função a ser reescrita pelo modelo.
type Task struct {
	Func     store.FuncRow
	Prompt   string
	Context  string // parte do Prompt antes do pedido: arquivo, imports, tipos, funções
	Doc      string // doc comment original, com quebra de linha no fim; pode ser vazio
	Original string // declaração original, sem o doc comment
	path     string // arquivo, absoluto
	lines    []string
}

// BuildTask monta a entrada do modelo para a função f usando DefaultCtxMode e DefaultBudgetChars.
func BuildTask(root string, db *sql.DB, f store.FuncRow) (*Task, error) {
	return BuildTaskWithMode(root, db, f, DefaultCtxMode, DefaultBudgetChars)
}

// BuildTaskWithMode monta a entrada do modelo para a função f no modo de
// contexto dado ("real", "oracle-distract" ou "oracle").
// No modo "real", absolutamente NADA do prompt depende do corpo da função-alvo:
// os imports do arquivo são avaliados com o corpo da função-alvo removido.
func BuildTaskWithMode(root string, db *sql.DB, f store.FuncRow, ctxMode string, budgetChars int) (*Task, error) {
	if ctxMode == "" {
		ctxMode = DefaultCtxMode
	}
	if budgetChars <= 0 {
		budgetChars = DefaultBudgetChars
	}
	path := filepath.Join(root, f.File)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(data), "\n")
	if f.LineStart < 1 || f.LineEnd > len(lines) || f.LineStart > f.LineEnd {
		return nil, fmt.Errorf("%s: linhas %d-%d fora do arquivo; rode o index de novo", f.File, f.LineStart, f.LineEnd)
	}
	t := &Task{Func: f, path: path, lines: lines, Original: strings.Join(lines[f.LineStart-1:f.LineEnd], "\n")}

	fset := token.NewFileSet()
	// Analisa o arquivo completo (com resolução sintática de objetos) para poder
	// esvaziar o corpo da função-alvo ao verificar quais imports existem fora dele.
	file, err := parser.ParseFile(fset, path, data, 0)
	if err != nil {
		return nil, err
	}

	if ctxMode == CtxReal {
		for _, d := range file.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fset.Position(fd.Pos()).Line == f.LineStart {
				if fd.Body != nil {
					fd.Body.List = nil
				}
				break
			}
		}
	}

	activeImports := map[string]bool{}
	var importLines []string
	for _, imp := range file.Imports {
		impPath := strings.Trim(imp.Path.Value, `"`)
		keep := true
		if ctxMode == CtxReal {
			if imp.Name != nil && (imp.Name.Name == "_" || imp.Name.Name == ".") {
				keep = true
			} else {
				keep = astutil.UsesImport(file, impPath)
			}
		}
		if keep {
			activeImports[impPath] = true
			line := imp.Path.Value
			if imp.Name != nil {
				line = imp.Name.Name + " " + line
			}
			importLines = append(importLines, line)
		}
	}

	var prefixBuf strings.Builder
	fmt.Fprintf(&prefixBuf, "Arquivo: %s\nPacote: %s\n", filepath.Base(f.File), file.Name.Name)
	if len(importLines) > 0 {
		prefixBuf.WriteString("\nImports:\n")
		for _, il := range importLines {
			prefixBuf.WriteString(il + "\n")
		}
	}

	first := f.LineStart - 1
	for first > 0 && strings.HasPrefix(strings.TrimSpace(lines[first-1]), "//") {
		first--
	}
	for _, l := range lines[first : f.LineStart-1] {
		t.Doc += l + "\n"
	}

	const maxDocChars = 6000
	hardCap := ctxMode == CtxReal || ctxMode == CtxOracleDistract
	maxDoc := maxDocChars
	if hardCap && budgetChars > 0 {
		baseFooter := "\nEscreva a função:\n```go\n" + f.Signature + "\n```\n"
		rem := budgetChars - prefixBuf.Len() - len(baseFooter)
		if rem < maxDoc {
			maxDoc = rem
		}
		if maxDoc < 0 {
			maxDoc = 0
		}
	}
	t.Doc = trimDocComment(t.Doc, maxDoc)
	footer := "\nEscreva a função:\n```go\n" + t.Doc + f.Signature + "\n```\n"

	ctxBudget := budgetChars
	if hardCap && budgetChars > 0 {
		ctxBudget = budgetChars - prefixBuf.Len() - len(footer)
		if ctxBudget < 0 {
			ctxBudget = 0
		}
	}

	mc := loadModCtx(root)
	var info funcContextInfo
	switch ctxMode {
	case CtxOracleDistract:
		info = mc.extractOracleContext(path, f.LineStart, true, ctxBudget)
	case CtxOracle:
		info = mc.extractOracleContext(path, f.LineStart, false, budgetChars)
	default:
		info = mc.extractRealContext(path, f.LineStart, activeImports, ctxBudget)
	}

	var b strings.Builder
	b.WriteString(prefixBuf.String())

	appendBlock := func(header, decl string) {
		if decl == "" {
			return
		}
		block := header + "```go\n" + decl + "\n```\n"
		if !hardCap || budgetChars <= 0 || b.Len()+len(block)+len(footer) <= budgetChars {
			b.WriteString(block)
			return
		}
		avail := budgetChars - b.Len() - len(footer) - len(header) - len("```go\n\n```\n")
		if avail <= 0 {
			return
		}
		cut := decl
		if len(cut) > avail {
			cut = cut[:avail]
			if nl := strings.LastIndexByte(cut, '\n'); nl > 0 {
				cut = cut[:nl]
			}
		}
		if strings.TrimSpace(cut) != "" {
			b.WriteString(header + "```go\n" + cut + "\n```\n")
		}
	}

	appendBlock("\nTipos:\n", info.typesDecl)
	appendBlock("\nConstantes e variáveis do pacote:\n", info.valuesDecl)

	avail, err := enrichFuncEntries(db, info.funcEntries)
	if err != nil {
		return nil, err
	}
	if len(avail) > 0 {
		header := "\nFunções disponíveis no projeto:\n"
		var kept []declEntry
		curLen := b.Len() + len(header) + len(footer)
		for _, fe := range avail {
			lineLen := len("- ") + len(fe.text) + 1
			if hardCap && budgetChars > 0 && curLen+lineLen > budgetChars {
				continue
			}
			kept = append(kept, fe)
			curLen += lineLen
		}
		// Ordena estritamente por ordem de declaração (arquivo e linha)
		sort.Slice(kept, func(i, j int) bool { return kept[i].pos < kept[j].pos })
		if len(kept) > 0 {
			b.WriteString(header)
			for _, fe := range kept {
				b.WriteString("- " + fe.text + "\n")
			}
		}
	}

	t.Context = b.String()
	t.Prompt = t.Context + footer
	return t, nil
}

// EditSystemPrompt é a instrução da tarefa de alteração.
const EditSystemPrompt = "Você altera funções Go conforme o pedido. Responda só com o código completo da função alterada, sem explicação."

// RequestPrompt é a entrada da escrita a partir de um pedido em texto: o
// pedido ocupa o lugar do doc comment.
func (t *Task) RequestPrompt(request string) string {
	return t.Context + "\nPedido: " + strings.TrimSpace(request) + "\n\nEscreva a função:\n```go\n" + t.Func.Signature + "\n```\n"
}

// EditPrompt é a entrada da alteração: a função como está e o que mudar nela.
func (t *Task) EditPrompt(request string) string {
	return t.Context + "\nFunção atual:\n```go\n" + t.Doc + t.Original + "\n```\n\nPedido: " + strings.TrimSpace(request) + "\n"
}

// enrichFuncEntries anexa a descrição curta (purpose) do banco SQLite às
// entradas de função, preservando a ordem recebida (para corte por prioridade
// antes da ordenação final por declaração).
func enrichFuncEntries(db *sql.DB, entries []declEntry) ([]declEntry, error) {
	if len(entries) == 0 {
		return nil, nil
	}
	descBySig := map[string]string{}
	if db != nil {
		rows, err := db.Query(`SELECT signature, COALESCE(description, '') FROM functions WHERE external = 0 AND signature <> '' AND description <> ''`)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		for rows.Next() {
			var sig, desc string
			if err := rows.Scan(&sig, &desc); err != nil {
				return nil, err
			}
			if p := describe.Purpose(desc); p != "" {
				descBySig[sig] = p
			}
		}
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}
	var out []declEntry
	for _, fe := range entries {
		sig := fe.text
		cleanSig := sig
		if idx := strings.Index(sig, ": func "); idx >= 0 {
			cleanSig = sig[idx+2:]
		}
		if p := descBySig[cleanSig]; p != "" {
			sig += " — " + p
		}
		feCopy := fe
		feCopy.text = sig
		out = append(out, feCopy)
	}
	return out, nil
}

// alignRE casa o alinhamento que o gofmt põe dentro de um bloco de declarações.
var alignRE = regexp.MustCompile(`[ \t]{2,}`)

// trimDocComment aplica o teto de maxChars ao doc comment da função-alvo:
// passo a) remove blocos de código indentados ("    // (exemplo omitido)");
// passo b) corta no fim de frase ou parágrafo ("// ... (documentação cortada)").
func trimDocComment(doc string, maxChars int) string {
	if len(doc) <= maxChars {
		return doc
	}
	if maxChars <= 0 {
		return ""
	}
	// Passo a: remover blocos indentados do doc Go
	lines := strings.Split(strings.TrimSuffix(doc, "\n"), "\n")
	isCodeLine := func(l string) bool {
		t := strings.TrimSpace(l)
		if !strings.HasPrefix(t, "//") {
			return false
		}
		after := strings.TrimPrefix(t, "//")
		// Go doc: linha de código começa com tab ou >= 2 espaços após "// " (ou >= 4 após "//")
		return strings.HasPrefix(after, "\t") || strings.HasPrefix(after, "    ") || strings.HasPrefix(after, " \t")
	}
	var out []string
	inCode := false
	for _, l := range lines {
		if isCodeLine(l) {
			if !inCode {
				out = append(out, "    // (exemplo omitido)")
				inCode = true
			}
			continue
		}
		inCode = false
		out = append(out, l)
	}
	stepA := strings.Join(out, "\n") + "\n"
	if len(stepA) <= maxChars {
		return stepA
	}

	// Passo b: cortar no fim de uma frase ou parágrafo sem quebrar frase
	const suffix = "// ... (documentação cortada)\n"
	limit := maxChars - len(suffix)
	if limit <= 0 {
		if len(suffix) <= maxChars {
			return suffix
		}
		return ""
	}
	prefix := stepA[:limit]
	best := -1
	for _, sep := range []string{".\n", "!\n", "?\n", ". ", "! ", "? ", "//\n", "// \n"} {
		if idx := strings.LastIndex(prefix, sep); idx >= 0 {
			end := idx + len(sep)
			if sep == ". " || sep == "! " || sep == "? " {
				// garante quebra de linha antes do sufixo
				end = idx + 1
			}
			if end > best {
				best = end
			}
		}
	}
	if best > 0 {
		res := strings.TrimRight(prefix[:best], " \t")
		if !strings.HasSuffix(res, "\n") {
			res += "\n"
		}
		return res + suffix
	}
	return suffix
}

func clip(text string, max int) string {
	if ls := strings.Split(text, "\n"); len(ls) > max {
		return strings.Join(ls[:max], "\n") + "\n\t// ..."
	}
	return text
}

// ExtractCode tira da resposta do modelo o código Go: o primeiro bloco
// cercado, se houver, a partir do primeiro "func".
func ExtractCode(answer string) string {
	if i := strings.Index(answer, "```"); i >= 0 {
		rest := answer[i+3:]
		if nl := strings.IndexByte(rest, '\n'); nl >= 0 {
			rest = rest[nl+1:]
		}
		if j := strings.Index(rest, "```"); j >= 0 {
			rest = rest[:j]
		}
		answer = rest
	}
	if i := strings.Index(answer, "func "); i > 0 {
		answer = answer[i:]
	}
	return strings.TrimSpace(answer)
}

// stdlib mapeia o nome usado no código para o caminho de import, para os
// pacotes padrão mais comuns. Serve para completar imports que faltarem.
var stdlib = map[string]string{
	"bufio": "bufio", "bytes": "bytes", "context": "context", "errors": "errors", "fmt": "fmt",
	"io": "io", "log": "log", "math": "math", "os": "os", "reflect": "reflect", "regexp": "regexp",
	"slices": "slices", "maps": "maps", "sort": "sort", "strconv": "strconv", "strings": "strings",
	"sync": "sync", "time": "time", "unicode": "unicode", "atomic": "sync/atomic",
	"filepath": "path/filepath", "path": "path", "json": "encoding/json", "hex": "encoding/hex",
	"base64": "encoding/base64", "binary": "encoding/binary", "http": "net/http", "url": "net/url",
	"net": "net", "utf8": "unicode/utf8", "rand": "math/rand", "exec": "os/exec", "big": "math/big",
}

// Replace devolve o arquivo da tarefa com a função trocada por code. Imports
// que ficaram sem uso saem e os da biblioteca padrão que faltarem entram,
// como um editor faria ao salvar.
func (t *Task) Replace(code string) ([]byte, error) {
	var out []string
	out = append(out, t.lines[:t.Func.LineStart-1]...)
	out = append(out, code)
	out = append(out, t.lines[t.Func.LineEnd:]...)
	src := []byte(strings.Join(out, "\n"))

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, t.path, src, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("código inválido: %w", err)
	}
	for _, group := range astutil.Imports(fset, file) {
		for _, imp := range group {
			path := strings.Trim(imp.Path.Value, `"`)
			if imp.Name != nil && (imp.Name.Name == "_" || imp.Name.Name == ".") {
				continue
			}
			if !astutil.UsesImport(file, path) {
				name := ""
				if imp.Name != nil {
					name = imp.Name.Name
				}
				astutil.DeleteNamedImport(fset, file, name, path)
			}
		}
	}
	// Nomes já ocupados por um import (crypto/rand como rand, por exemplo)
	// não podem receber outro pacote.
	taken := map[string]bool{}
	for _, imp := range file.Imports {
		path := strings.Trim(imp.Path.Value, `"`)
		name := path[strings.LastIndexByte(path, '/')+1:]
		if imp.Name != nil {
			name = imp.Name.Name
		}
		taken[name] = true
	}
	for _, id := range file.Unresolved {
		if path, ok := stdlib[id.Name]; ok && !taken[id.Name] {
			astutil.AddImport(fset, file, path)
			taken[id.Name] = true
		}
	}
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, file); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// ReplaceWithGoImports devolve o arquivo da tarefa com a função trocada por
// code e processado via goimports (golang.org/x/tools/imports) para ajustar
// imports da stdlib, do módulo e de terceiros antes da compilação.
func (t *Task) ReplaceWithGoImports(code string) ([]byte, error) {
	var out []string
	out = append(out, t.lines[:t.Func.LineStart-1]...)
	out = append(out, code)
	out = append(out, t.lines[t.Func.LineEnd:]...)
	src := []byte(strings.Join(out, "\n"))
	formatted, err := imports.Process(t.path, src, nil)
	if err != nil {
		return nil, fmt.Errorf("código inválido: %w", err)
	}
	return formatted, nil
}
