package describe

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthropics/anthropic-sdk-go/v2"
)

// Backend gera o texto da descrição a partir das regras e da entrada.
type Backend interface {
	// Name identifica backend e modelo; vai para meta.model_version.
	Name() string
	Generate(ctx context.Context, system, user string) (string, error)
}

// Ollama usa um modelo local servido pelo Ollama.
type Ollama struct {
	Host  string // por exemplo http://localhost:11434
	Model string
	http  http.Client

	tunedOnce sync.Once
	tuned     bool
}

func NewOllama(host, model string) *Ollama {
	return &Ollama{Host: strings.TrimRight(host, "/"), Model: model, http: http.Client{Timeout: 10 * time.Minute}}
}

func (o *Ollama) Name() string { return "ollama/" + o.Model }

// isTuned diz se o modelo foi treinado para esta tarefa. O sinal é o prompt
// de sistema gravado no Modelfile por train/export.py, não o nome do modelo:
// assim o modelo pode se chamar como o usuário quiser.
func (o *Ollama) isTuned(ctx context.Context) bool {
	o.tunedOnce.Do(func() {
		reqBody, _ := json.Marshal(map[string]string{"model": o.Model})
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.Host+"/api/show", bytes.NewReader(reqBody))
		if err != nil {
			return
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := o.http.Do(req)
		if err != nil {
			return
		}
		defer resp.Body.Close()
		var out struct {
			System string `json:"system"`
		}
		if json.NewDecoder(resp.Body).Decode(&out) == nil {
			o.tuned = strings.TrimSpace(out.System) == TunedSystemPrompt
		}
	})
	return o.tuned
}

func (o *Ollama) Generate(ctx context.Context, system, user string) (string, error) {
	// num_predict é teto de segurança: sem ele, uma saída que entra em laço
	// gera até encher o contexto e prende a fila do Ollama por minutos.
	body := map[string]any{
		"model":   o.Model,
		"stream":  false,
		"think":   false,
		"format":  Schema(),
		"options": map[string]any{"temperature": 0, "num_ctx": 8192, "num_predict": 700},
	}
	if o.isTuned(ctx) {
		// O modelo treinado aprendeu o formato com os campos numa ordem fixa.
		// O esquema imporia outra ordem (alfabética) e o tiraria do que ele
		// viu no treino; a validação de Parse cobre os deslizes.
		system = TunedSystemPrompt
		delete(body, "format")
		body["options"] = map[string]any{"temperature": 0, "num_ctx": 4096, "num_predict": 400}
	}
	body["messages"] = []map[string]string{
		{"role": "system", "content": system},
		{"role": "user", "content": user},
	}
	reqBody, err := json.Marshal(body)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.Host+"/api/chat", bytes.NewReader(reqBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := o.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama: %s: %s", resp.Status, strings.TrimSpace(string(data)))
	}
	var out struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return "", fmt.Errorf("ollama: resposta inválida: %w", err)
	}
	return out.Message.Content, nil
}

// Anthropic usa a API da Anthropic (lê ANTHROPIC_API_KEY do ambiente).
type Anthropic struct {
	Model  string
	Effort string
	Format map[string]any // esquema JSON da resposta; nil = o da descrição
	client anthropic.Client

	inputTokens, outputTokens atomic.Int64
}

// Usage devolve os tokens de entrada e de saída consumidos até agora.
func (a *Anthropic) Usage() (input, output int64) {
	return a.inputTokens.Load(), a.outputTokens.Load()
}

func NewAnthropic(model, effort string) *Anthropic {
	return &Anthropic{Model: model, Effort: effort, client: anthropic.NewClient()}
}

func (a *Anthropic) Name() string { return "anthropic/" + a.Model }

func (a *Anthropic) Generate(ctx context.Context, system, user string) (string, error) {
	format := a.Format
	if format == nil {
		format = Schema()
	}
	resp, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     a.Model,
		MaxTokens: 16000,
		System:    []anthropic.TextBlockParam{{Text: system}},
		OutputConfig: anthropic.OutputConfigParam{
			Effort: anthropic.OutputConfigEffort(a.Effort),
			Format: anthropic.JSONOutputFormatParam{Schema: format},
		},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(user)),
		},
	})
	if err != nil {
		return "", err
	}
	a.inputTokens.Add(resp.Usage.InputTokens + resp.Usage.CacheReadInputTokens + resp.Usage.CacheCreationInputTokens)
	a.outputTokens.Add(resp.Usage.OutputTokens)
	switch resp.StopReason {
	case anthropic.StopReasonRefusal:
		return "", fmt.Errorf("o modelo recusou o pedido (%s)", resp.StopDetails.Category)
	case anthropic.StopReasonMaxTokens:
		return "", fmt.Errorf("resposta cortada em max_tokens")
	}
	var text strings.Builder
	for _, block := range resp.Content {
		if t, ok := block.AsAny().(anthropic.TextBlock); ok {
			text.WriteString(t.Text)
		}
	}
	return text.String(), nil
}
