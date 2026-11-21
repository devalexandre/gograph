package implement

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"regexp"
	"strconv"
	"strings"
)

// FixSystemPrompt é a instrução da tarefa de correção. Como SystemPrompt, o
// texto e o formato de FixPrompt precisam ser os mesmos no treino e no uso.
const FixSystemPrompt = "Você corrige funções Go. Responda só com o código completo da função corrigida, sem explicação."

const maxFixErrors = 6 // erros do compilador mostrados ao modelo

// FixPrompt monta a entrada da correção: o pedido original, a tentativa que
// não compilou e os erros do compilador.
func FixPrompt(prompt, code string, errs []string) string {
	var b strings.Builder
	b.WriteString(prompt)
	b.WriteString("\nTentativa que não compila:\n```go\n" + code + "\n```\n\nErros do compilador:\n")
	for _, e := range errs {
		b.WriteString("- " + e + "\n")
	}
	return b.String()
}

// TestFixPrompt monta a entrada da correção quando o código compilou mas
// falhou nos testes do pacote.
func TestFixPrompt(prompt, code, testOut string) string {
	var b strings.Builder
	b.WriteString(prompt)
	b.WriteString("\nTentativa que falhou nos testes:\n```go\n" + code + "\n```\n\nSaída dos testes:\n```\n" + testOut + "\n```\n")
	return b.String()
}

// ClassifyError classifica o primeiro erro de compilação/vet/teste em uma das
// categorias definidas na Etapa 3.
func ClassifyError(trigger, errStr string) string {
	if trigger == "length" {
		return "truncado"
	}
	if trigger == "test" {
		return "teste falhou"
	}
	if errStr == "" {
		return "outros"
	}
	if strings.HasPrefix(errStr, "resposta sem função") || strings.HasPrefix(errStr, "código inválido:") {
		return "erro de sintaxe"
	}
	msg := errStr
	if idx := strings.Index(msg, ": "); idx >= 0 && strings.HasPrefix(msg, "linha ") {
		msg = msg[idx+2:]
	} else if idx := strings.Index(msg, ": "); idx >= 0 && (strings.HasPrefix(msg, "em ") || strings.HasPrefix(msg, "fora da função: ")) {
		msg = msg[idx+2:]
	}
	switch {
	case strings.Contains(msg, "imported and not used") || strings.Contains(msg, "could not import") || strings.Contains(msg, "no required module provides"):
		return "import faltando ou não usado"
	case strings.Contains(msg, "declared and not used"):
		return "variável declarada e não usada"
	case strings.Contains(msg, "has no field or method") || strings.Contains(msg, "undefined (type "):
		return "campo ou método inexistente"
	case strings.Contains(msg, "undefined:"):
		sym := strings.TrimSpace(strings.TrimPrefix(msg[strings.Index(msg, "undefined:"):], "undefined:"))
		if sp := strings.IndexAny(sym, " \t\n("); sp >= 0 {
			sym = sym[:sp]
		}
		if _, ok := stdlib[sym]; ok || knownPkgNames[sym] {
			return "import faltando ou não usado"
		}
		return "símbolo indefinido"
	case strings.Contains(msg, "cannot use ") || strings.Contains(msg, "cannot convert ") ||
		strings.Contains(msg, "invalid operation:") || strings.Contains(msg, "cannot assign ") ||
		strings.Contains(msg, "cannot range over ") || strings.Contains(msg, "cannot index ") ||
		strings.Contains(msg, "mismatched types") || strings.Contains(msg, "does not implement") ||
		strings.Contains(msg, "cannot take the address") || strings.Contains(msg, "non-interface type") ||
		strings.Contains(msg, "impossible type assertion"):
		return "tipo incompatível"
	case strings.Contains(msg, "too many arguments") || strings.Contains(msg, "not enough arguments") ||
		strings.Contains(msg, "too many return values") || strings.Contains(msg, "not enough return values") ||
		strings.Contains(msg, "wrong number of") || strings.Contains(msg, "missing return") ||
		strings.Contains(msg, "assignment mismatch:"):
		return "número errado de argumentos/retornos"
	case strings.Contains(msg, "syntax error") || strings.Contains(msg, "expected ") || strings.Contains(msg, "unexpected "):
		return "erro de sintaxe"
	default:
		return "outros"
	}
}

var knownPkgNames = map[string]bool{
	"tea": true, "colorful": true, "uv": true, "ansi": true, "lipgloss": true, "term": true,
	"cancelreader": true, "colorprofile": true, "golden": true, "yaml": true, "toml": true,
	"jsonschema": true, "compose": true, "api": true, "errdefs": true, "errgroup": true,
	"logrus": true, "sirupsen": true, "otel": true, "trace": true, "attribute": true,
	"semconv": true, "containerd": true, "moby": true, "docker": true, "cli": true,
	"cobra": true, "pflag": true, "mux": true, "gorilla": true, "jwt": true, "uuid": true,
	"pkcs12": true, "mscfb": true, "efp": true, "image": true, "freetype": true, "truetype": true,
	"selinux": true, "git": true, "gh": true, "actions": true, "model": true, "runner": true,
	"common": true, "expr": true, "schema": true, "dockerclient": true, "viper": true,
	"cast": true, "spew": true, "require": true, "assert": true, "testify": true,
	"units": true, "archive": true, "digest": true, "shellquote": true, "deepcopy": true,
	"paths": true, "formatter": true, "display": true, "prompt": true, "desktop": true,
	"inspect": true, "logstream": true, "oci": true, "projectloader": true, "watch": true,
	"dryrun": true, "e2e": true, "utils": true, "artifacts": true, "artifactcache": true,
	"exprparser": true, "container": true, "list": true, "heap": true, "ring": true,
	"xml": true, "csv": true, "zip": true, "gzip": true, "tar": true, "hash": true,
	"sha256": true, "md5": true, "tls": true, "x509": true, "pem": true, "rsa": true,
	"ecdsa": true, "ed25519": true, "asn1": true, "elliptic": true, "pkix": true,
	"ioutil": true, "embed": true, "fs": true, "cmp": true, "iter": true, "runtime": true,
	"syscall": true, "unsafe": true, "flag": true, "testing": true, "html": true, "template": true,
}

var errLineRE = regexp.MustCompile(`^(?:\S*/)?([^/\s:]+\.go):(\d+)(?::\d+)?: (.+)$`)

// compilerErrors traduz a saída do compilador para a visão do modelo: ele só
// conhece a função, então a linha é contada a partir do início dela (start, na
// numeração do arquivo own) e o caminho do arquivo some.
func compilerErrors(out, own string, start, lines int) []string {
	var errs []string
	for _, line := range strings.Split(out, "\n") {
		m := errLineRE.FindStringSubmatch(strings.TrimPrefix(line, "código inválido: "))
		if m == nil {
			// "have (...)" e "want (...)" completam o erro anterior.
			if len(errs) > 0 && strings.HasPrefix(line, "\t") {
				errs[len(errs)-1] += "; " + strings.TrimSpace(line)
			}
			continue
		}
		if len(errs) == maxFixErrors {
			break
		}
		n, _ := strconv.Atoi(m[2])
		switch rel := n - start + 1; {
		case m[1] != own:
			errs = append(errs, "em "+m[1]+": "+m[3])
		case rel < 1 || rel > lines:
			errs = append(errs, "fora da função: "+m[3])
		default:
			errs = append(errs, fmt.Sprintf("linha %d: %s", rel, m[3]))
		}
	}
	return errs
}

func lineCount(s string) int { return strings.Count(s, "\n") + 1 }

// locate acha em src (o arquivo já com a função trocada e formatado) a função
// que code declara. Devolve o texto dela como ficou no arquivo e a linha em
// que começa: é sobre esse texto que o compilador numera os erros.
func locate(src []byte, code string) (string, int) {
	want := firstFunc(token.NewFileSet(), []byte("package p\n"+code))
	if want == nil {
		return "", 0
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", src, parser.SkipObjectResolution)
	if err != nil {
		return "", 0
	}
	for _, d := range file.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Name.Name != want.Name.Name || recvName(fd) != recvName(want) {
			continue
		}
		from, to := fset.Position(fd.Pos()), fset.Position(fd.End())
		return string(src[from.Offset:to.Offset]), from.Line
	}
	return "", 0
}

func firstFunc(fset *token.FileSet, src []byte) *ast.FuncDecl {
	file, err := parser.ParseFile(fset, "", src, parser.SkipObjectResolution)
	if err != nil {
		return nil
	}
	for _, d := range file.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok {
			return fd
		}
	}
	return nil
}

// recvName devolve o nome do tipo do receptor, sem ponteiro nem parâmetros de tipo.
func recvName(fd *ast.FuncDecl) string {
	if fd.Recv == nil || len(fd.Recv.List) == 0 {
		return ""
	}
	e := fd.Recv.List[0].Type
	for {
		switch t := e.(type) {
		case *ast.StarExpr:
			e = t.X
		case *ast.ParenExpr:
			e = t.X
		case *ast.IndexExpr:
			e = t.X
		case *ast.IndexListExpr:
			e = t.X
		case *ast.Ident:
			return t.Name
		default:
			return ""
		}
	}
}

// FixRecord transforma uma tentativa que não compilou em exemplo de treino: a
// entrada é a de FixPrompt e a saída é a função original.
func FixRecord(r Result, source, license string) (Record, bool) {
	if r.Built || len(r.Errors) == 0 || r.Code == "" || r.Original == "" {
		return Record{}, false
	}
	return Record{
		Source: source, License: license, Name: r.Name, Hash: r.Hash,
		System: FixSystemPrompt, Input: FixPrompt(r.Prompt, r.Code, r.Errors), Output: r.Original,
	}, true
}
