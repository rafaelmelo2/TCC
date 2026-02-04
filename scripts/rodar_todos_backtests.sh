#!/usr/bin/env bash
#
# Roda backtests para os 3 ativos × 5 folds × 2 estratégias (long_short, long_only).
# Resultados salvos em data/backtest/ com nome detalhado.
#
# Uso (a partir do diretório do pipeline):
#   ./scripts/rodar_todos_backtests.sh
# ou:
#   bash scripts/rodar_todos_backtests.sh
#

set -e
cd "$(dirname "$0")/.."

ATIVOS=(PETR4 VALE3 ITUB4)
FOLDS=(1 2 3 4 5)
ESTRATEGIAS=(long_short long_only)

total=$(( ${#ATIVOS[@]} * ${#FOLDS[@]} * ${#ESTRATEGIAS[@]} ))
n=0

echo "========================================"
echo "RODANDO TODOS OS BACKTESTS"
echo "========================================"
echo "Ativos: ${ATIVOS[*]}"
echo "Folds: ${FOLDS[*]}"
echo "Estratégias: ${ESTRATEGIAS[*]}"
echo "Total: $total execuções"
echo "========================================"

for ativo in "${ATIVOS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    for estrategia in "${ESTRATEGIAS[@]}"; do
      n=$(( n + 1 ))
      echo ""
      echo "[$n/$total] $ativo fold $fold $estrategia"
      uv run python src/scripts/rodar_backtest.py --ativo "$ativo" --fold "$fold" --estrategia "$estrategia" || true
    done
  done
done

echo ""
echo "========================================"
echo "CONCLUÍDO. Resultados em data/backtest/"
echo "========================================"
