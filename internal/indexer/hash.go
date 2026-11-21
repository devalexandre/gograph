package indexer

import (
	"crypto/sha256"
	"encoding/hex"
	"go/scanner"
	"go/token"
)

// hashSource calcula o SHA-256 da sequência de tokens de um trecho de código
// Go. Comentários e espaços não entram, então gofmt e mudança de comentário
// não alteram o hash.
//
// O plano previa imprimir o FuncDecl com go/printer sem comentários, mas o
// printer preserva linhas em branco conforme as posições originais: apagar um
// comentário entre duas instruções mudaria a saída. Tokens não têm esse
// problema.
func hashSource(src []byte) string {
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(src))
	var s scanner.Scanner
	s.Init(file, src, nil, 0) // sem ScanComments: comentários são ignorados

	h := sha256.New()
	pendingSemi := false
	for {
		_, tok, lit := s.Scan()
		if tok == token.EOF {
			break
		}
		// Ponto e vírgula antes de "}" ou ")" depende só da quebra de linha
		// ("{ return x }" contra a forma em várias linhas), então é adiado e
		// descartado nesses casos.
		if tok == token.SEMICOLON {
			pendingSemi = true
			continue
		}
		if pendingSemi {
			if tok != token.RBRACE && tok != token.RPAREN {
				h.Write([]byte{';', 0})
			}
			pendingSemi = false
		}
		if lit == "" {
			lit = tok.String()
		}
		h.Write([]byte(lit))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}
