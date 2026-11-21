#!/usr/bin/env bash
# Grava as tarefas de escrita dos 6 repositórios de teste, para um gerador
# externo (train/gerar.py) produzir as respostas.
#
# Uso: ./bench/tarefas.sh <saida.jsonl> [n] [-tail]
#
#   n      funções por repositório (padrão 60, a amostra da medição)
#   -tail  a outra ponta da amostra: funções de VALIDAÇÃO, que a medição
#          final não usa. Escolher checkpoint olhando o próprio conjunto de
#          teste inflaria o resultado.
set -euo pipefail
# O caminho de saída vale a partir de onde o script foi chamado.
saida=$(realpath -m "$1")
cd "$(dirname "$0")/.."

n=${2:-60}
shift $(( $# > 1 ? 2 : 1 ))
GO=${GO:-$HOME/.gvm/gos/go1.27.0/bin/go}
BIN=${BIN:-./bin/gograph}
REPOS=${REPOS:-"gorilla__mux charmbracelet__bubbletea docker__compose nektos__act FiloSottile__mkcert qax-os__excelize"}

: > "$saida"
for r in $REPOS; do
  "$BIN" bench -db "corpus/db/$r.db" -corpus corpus -go "$GO" -n "$n" -dump -out "$saida.parte" "$@" "corpus/src/$r" >/dev/null
  cat "$saida.parte" >> "$saida"
done
rm -f "$saida.parte"
echo "$(wc -l < "$saida") tarefas em $saida"
