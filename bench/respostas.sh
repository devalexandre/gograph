#!/usr/bin/env bash
# Confere, com o compilador, respostas geradas fora do Ollama.
#
# Uso: ./bench/respostas.sh <respostas.jsonl> [n] [opções do gograph bench]
#
#   ./bench/respostas.sh val.jsonl 20 -tail          validação, só compilação
#   PREFIXO=hf-base ./bench/respostas.sh r.jsonl 60 -tests
#                                                    medição final; grava
#                                                    bench/hf-base-<repo>.jsonl
#
# Imprime "compilam: X de Y" por repositório e o total. Com -tests roda go test
# em código gerado pelo modelo: só nos repositórios open source do corpus.
set -euo pipefail
# O caminho das respostas vale a partir de onde o script foi chamado.
respostas=$(realpath "$1")
cd "$(dirname "$0")/.."

n=${2:-60}
shift $(( $# > 1 ? 2 : 1 ))
GO=${GO:-$HOME/.gvm/gos/go1.27.0/bin/go}
BIN=${BIN:-./bin/gograph}
REPOS=${REPOS:-"gorilla__mux charmbracelet__bubbletea docker__compose nektos__act FiloSottile__mkcert qax-os__excelize"}

total=0
feitas=0
for r in $REPOS; do
  saida=()
  if [ -n "${PREFIXO:-}" ]; then
    saida=(-out "bench/$PREFIXO-$r.jsonl")
  fi
  linhas=$("$BIN" bench -db "corpus/db/$r.db" -corpus corpus -go "$GO" -n "$n" -answers "$respostas" "${saida[@]}" "$@" "corpus/src/$r" 2>&1 | grep -E "^(compilam|testes):" || true)
  echo "== $r"
  echo "$linhas"
  c=$(echo "$linhas" | sed -n 's/^compilam: *\([0-9]*\) de \([0-9]*\).*/\1 \2/p')
  feitas=$((feitas + ${c%% *}))
  total=$((total + ${c##* }))
done
echo "total: $feitas de $total compilam"
