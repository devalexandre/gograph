// Package describe gera a descrição local de cada função do grafo.
package describe

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
)

// Limites do formato da descrição.
const (
	MaxPurposeWords = 20 // limite pedido ao modelo
	MaxCalls        = 5

	// purposeTolerance é o que o validador aceita. Os modelos contam palavras
	// de forma aproximada e passam do limite por uma ou duas; rejeitar isso
	// só gasta uma segunda chamada.
	purposeTolerance = 25
)

// Description é a descrição de uma função. Fala só do que a função faz
// localmente; o fluxo completo vem do grafo.
type Description struct {
	Purpose     string   `json:"purpose"`
	Calls       []Call   `json:"calls"`
	Effects     []string `json:"effects"`
	Errors      []string `json:"errors"`
	Concurrency string   `json:"concurrency"`
}

// Call cita uma chamada relevante e a intenção dela.
type Call struct {
	Fn  string `json:"fn"`
	Why string `json:"why"`
}

// Schema é o JSON Schema da descrição, usado para restringir a saída do modelo.
func Schema() map[string]any {
	str := map[string]any{"type": "string"}
	strList := map[string]any{"type": "array", "items": str}
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"purpose": str,
			"calls": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type":                 "object",
					"properties":           map[string]any{"fn": str, "why": str},
					"required":             []string{"fn", "why"},
					"additionalProperties": false,
				},
			},
			"effects":     strList,
			"errors":      strList,
			"concurrency": str,
		},
		"required":             []string{"purpose", "calls", "effects", "errors", "concurrency"},
		"additionalProperties": false,
	}
}

// Parse valida a saída do modelo contra o formato e devolve a descrição
// normalizada. allowed são os nomes de callee que podem aparecer em calls.
func Parse(raw string, allowed map[string]bool) (*Description, error) {
	raw = stripFence(raw)

	var d Description
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&d); err != nil {
		return nil, fmt.Errorf("JSON inválido: %w", err)
	}
	d.Purpose = strings.TrimSpace(d.Purpose)
	if d.Purpose == "" {
		return nil, fmt.Errorf("purpose vazio")
	}
	if n := len(strings.Fields(d.Purpose)); n > purposeTolerance {
		return nil, fmt.Errorf("purpose tem %d palavras; o máximo é %d", n, MaxPurposeWords)
	}
	if len(d.Calls) > MaxCalls {
		return nil, fmt.Errorf("calls tem %d itens; o máximo é %d", len(d.Calls), MaxCalls)
	}
	for _, c := range d.Calls {
		if !allowed[c.Fn] {
			return nil, &UnknownCallError{Fn: c.Fn}
		}
		if strings.TrimSpace(c.Why) == "" {
			return nil, fmt.Errorf("calls: %q sem intenção (why)", c.Fn)
		}
	}
	d.Errors = dropNegations(d.Errors)
	if d.Calls == nil {
		d.Calls = []Call{}
	}
	if d.Effects == nil {
		d.Effects = []string{}
	}
	if d.Errors == nil {
		d.Errors = []string{}
	}
	return &d, nil
}

func stripFence(raw string) string {
	raw = strings.TrimSpace(raw)
	raw = strings.TrimPrefix(raw, "```json")
	raw = strings.TrimPrefix(raw, "```")
	return strings.TrimSuffix(raw, "```")
}

// UnknownCallError é a rejeição por calls citar uma função que não está entre
// as chamadas reais da função descrita.
type UnknownCallError struct{ Fn string }

func (e *UnknownCallError) Error() string {
	return fmt.Sprintf("calls cita %q, que não está na lista de chamadas da função", e.Fn)
}

// Repair corrige o que dá para corrigir sem o modelo: tira de calls as funções
// que o grafo não confirma, mantendo o resto da descrição. Serve aos modelos
// locais, onde uma segunda chamada custa tempo e raramente muda o resultado.
func Repair(raw string, allowed map[string]bool) (*Description, error) {
	d, err := Parse(raw, allowed)
	var unknown *UnknownCallError
	if !errors.As(err, &unknown) {
		return d, err
	}
	var loose Description
	if err := json.Unmarshal([]byte(stripFence(raw)), &loose); err != nil {
		return nil, err
	}
	kept := loose.Calls[:0]
	for _, c := range loose.Calls {
		if allowed[c.Fn] {
			kept = append(kept, c)
		}
	}
	loose.Calls = kept
	fixed, err := json.Marshal(loose)
	if err != nil {
		return nil, err
	}
	return Parse(string(fixed), allowed)
}

// negation reconhece itens de errors que só dizem que não há erro. O modelo
// professor escreve isso com frequência e os modelos treinados herdaram; a
// lista vazia já diz a mesma coisa.
var negation = regexp.MustCompile(`(?i)^(não (devolve|retorna|há|gera|produz)|nenhum|sem) erro`)

func dropNegations(errs []string) []string {
	kept := errs[:0]
	for _, e := range errs {
		if !negation.MatchString(strings.TrimSpace(e)) {
			kept = append(kept, e)
		}
	}
	return kept
}

// Purpose extrai o campo purpose de uma descrição gravada.
func Purpose(descriptionJSON string) string {
	if descriptionJSON == "" {
		return ""
	}
	var d Description
	if json.Unmarshal([]byte(descriptionJSON), &d) != nil {
		return ""
	}
	return d.Purpose
}
