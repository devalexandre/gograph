package implement

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/devalexandre/gograph/internal/store"
)

// TestNoLeakInCtxReal verifica que, para 30 funções do benchmark (5 de cada um
// dos 6 repositórios, incluindo qax-os/excelize e nektos/act), o prompt gerado
// em ctx=real com o arquivo original é IDÊNTICO ao prompt gerado quando o corpo
// da função-alvo no arquivo é substituído por panic("TODO"), tanto com
// --budget=6000 quanto com --budget=24000.
func TestNoLeakInCtxReal(t *testing.T) {
	root, err := filepath.Abs("../../corpus")
	if err != nil {
		t.Fatal(err)
	}
	repos := []string{
		"gorilla__mux",
		"charmbracelet__bubbletea",
		"FiloSottile__mkcert",
		"docker__compose",
		"qax-os__excelize",
		"nektos__act",
	}
	budgets := []int{6000, 24000}

	for _, budget := range budgets {
		budget := budget
		t.Run(fmt.Sprintf("budget=%d", budget), func(t *testing.T) {
			var tested int
			for _, repo := range repos {
				srcDir := filepath.Join(root, "src", repo)
				dbPath := filepath.Join(root, "db", repo+".db")
				if _, err := os.Stat(dbPath); err != nil {
					t.Skip("corpus ausente:", dbPath)
				}
				st, err := store.Open(dbPath)
				if err != nil {
					t.Fatal(err)
				}
				funcs, err := candidates(st, srcDir, 5, 5, 40, false)
				if err != nil {
					st.Close()
					t.Fatal(err)
				}
				if len(funcs) < 5 {
					st.Close()
					t.Fatalf("%s: esperava 5 funções candidatas, encontrou %d", repo, len(funcs))
				}

				for _, fn := range funcs[:5] {
					// 1. Gera prompt a partir do arquivo original
					InvalidateModCtx()
					origTask, err := BuildTaskWithMode(srcDir, st.DB(), fn, CtxReal, budget)
					if err != nil {
						st.Close()
						t.Fatalf("%s: BuildTask original falhou: %v", fn.Name, err)
					}

					// 2. Substitui o corpo da função-alvo por panic("TODO") no arquivo
					filePath := filepath.Join(srcDir, fn.File)
					origBytes, err := os.ReadFile(filePath)
					if err != nil {
						st.Close()
						t.Fatal(err)
					}

					stubCode := fn.Signature + " {\n\tpanic(\"TODO\")\n}"
					stubBytes, err := origTask.Replace(stubCode)
					if err != nil {
						st.Close()
						t.Fatalf("%s: Replace por panic(\"TODO\") falhou: %v", fn.Name, err)
					}

					// Ajusta LineStart / LineEnd caso Replace tenha reformatado os imports acima da função
					stubFn := fn
					_, newStart := locate(stubBytes, stubCode)
					if newStart > 0 {
						stubFn.LineStart = newStart
						stubFn.LineEnd = newStart + strings.Count(stubCode, "\n")
					}

					if err := os.WriteFile(filePath, stubBytes, 0o644); err != nil {
						st.Close()
						t.Fatal(err)
					}

					InvalidateModCtx()
					stubTask, stubErr := BuildTaskWithMode(srcDir, st.DB(), stubFn, CtxReal, budget)

					// Restaura imediatamente o arquivo original
					_ = os.WriteFile(filePath, origBytes, 0o644)
					InvalidateModCtx()

					if stubErr != nil {
						st.Close()
						t.Fatalf("%s: BuildTask com panic(\"TODO\") falhou: %v", fn.Name, stubErr)
					}

					if origTask.Prompt != stubTask.Prompt {
						st.Close()
						t.Fatalf("VAZAMENTO DETECTADO (budget=%d) em %s (%s):\n--- ORIGINAL ---\n%s\n--- COM PANIC(TODO) ---\n%s",
							budget, fn.Name, repo, origTask.Prompt, stubTask.Prompt)
					}
					tested++
				}
				st.Close()
			}
			if tested != 30 {
				t.Fatalf("esperava testar 30 funções do benchmark, testou %d", tested)
			}
			t.Logf("SUCESSO (budget=%d): 30/30 funções do benchmark (5 por repositório) geraram prompts 100%% idênticos entre o código original e panic(\"TODO\") em ctx=real", budget)
		})
	}
}
