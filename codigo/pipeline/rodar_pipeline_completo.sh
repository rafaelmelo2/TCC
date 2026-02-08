#!/usr/bin/env bash
# ============================================================================
# Pipeline Completo do TCC
# Predição Automática de Indicativos Financeiros para Bolsa de Valores
# Considerando o Aspecto Temporal
#
# Gera TODOS os artefatos (modelos, métricas, backtests, testes estatísticos,
# visualizações) do zero, para os 3 ativos: PETR4, VALE3, ITUB4.
#
# Resultados antigos são movidos para backup/ com timestamp.
#
# Uso:
#   chmod +x rodar_pipeline_completo.sh
#   ./rodar_pipeline_completo.sh              # pipeline completo
#   ./rodar_pipeline_completo.sh --sem-gpu    # forçar CPU
#   ./rodar_pipeline_completo.sh --rapido     # menos epochs (para testar)
# ============================================================================
set -euo pipefail

# ── Configuração ────────────────────────────────────────────────────────────
ATIVOS=("PETR4" "VALE3" "ITUB4")
MODELO="cnn_lstm"
EPOCHS=100               # epochs por fold (early stopping corta antes)
OPTUNA_TRIALS=20          # trials do Optuna para hiperparâmetros
N_FOLDS=5                 # número de folds esperados
GPU_FLAG="--gpu"          # padrão: usar GPU
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${PIPELINE_DIR}/logs/pipeline"
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# ── Argumentos ──────────────────────────────────────────────────────────────
for arg in "$@"; do
    case $arg in
        --sem-gpu)    GPU_FLAG="--no-gpu" ;;
        --rapido)     EPOCHS=15; OPTUNA_TRIALS=5 ;;
        *)            echo "Argumento desconhecido: $arg"; exit 1 ;;
    esac
done

# ── Funções utilitárias ────────────────────────────────────────────────────
verde()    { echo -e "\033[0;32m$*\033[0m"; }
amarelo()  { echo -e "\033[0;33m$*\033[0m"; }
vermelho() { echo -e "\033[0;31m$*\033[0m"; }

etapa() {
    echo ""
    echo "======================================================================"
    verde "  $1"
    echo "======================================================================"
    echo ""
}

rodar() {
    # Executa comando, imprime no terminal e no log
    echo "[$(date +%H:%M:%S)] CMD: $*" | tee -a "$LOG_FILE"
    if ! uv run python "$@" 2>&1 | tee -a "$LOG_FILE"; then
        amarelo "[AVISO] Comando falhou (continuando): $*"
    fi
}

tempo_decorrido() {
    local inicio=$1
    local agora
    agora=$(date +%s)
    local diff=$((agora - inicio))
    printf '%02dh:%02dm:%02ds' $((diff/3600)) $((diff%3600/60)) $((diff%60))
}

# ── Início ──────────────────────────────────────────────────────────────────
INICIO_GLOBAL=$(date +%s)
mkdir -p "$LOG_DIR"

etapa "PIPELINE COMPLETO DO TCC – ${TIMESTAMP}"
echo "Diretório:  ${PIPELINE_DIR}"
echo "Ativos:     ${ATIVOS[*]}"
echo "Modelo:     ${MODELO}"
echo "Epochs:     ${EPOCHS}"
echo "GPU:        ${GPU_FLAG}"
echo "Log:        ${LOG_FILE}"
echo ""

cd "$PIPELINE_DIR"

# ============================================================================
# ETAPA 0 – BACKUP DOS DADOS ANTIGOS
# ============================================================================
etapa "ETAPA 0/7 – Backup dos dados antigos"

BACKUP_DIR="backup/${TIMESTAMP}"
mkdir -p "$BACKUP_DIR"

for dir in data/processed data/backtest models logs/training_history; do
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "  Movendo ${dir}/ → ${BACKUP_DIR}/${dir}/"
        mkdir -p "${BACKUP_DIR}/$(dirname "$dir")"
        cp -r "$dir" "${BACKUP_DIR}/${dir}"
    fi
done
echo ""
verde "[OK] Backup salvo em: ${BACKUP_DIR}/"

# Limpar diretórios de saída (manter raw!)
rm -rf data/processed data/backtest models logs/training_history
mkdir -p data/processed data/backtest models logs/training_history

# ============================================================================
# ETAPA 1 – VISUALIZAÇÃO DE FEATURES
# ============================================================================
etapa "ETAPA 1/7 – Visualização de features (todos os ativos)"

for ATIVO in "${ATIVOS[@]}"; do
    echo "── ${ATIVO} ──"
    rodar src/scripts/visualizar_features.py --ativo "$ATIVO"
done

# ============================================================================
# ETAPA 2 – BASELINES (Naive, Drift, ARIMA, Prophet)
# ============================================================================
etapa "ETAPA 2/7 – Treinamento dos baselines (walk-forward)"

rodar src/tests/testar_baselines_walkforward.py --todos

# ============================================================================
# ETAPA 3 – TREINAMENTO CNN-LSTM (walk-forward + Optuna por fold)
# ============================================================================
etapa "ETAPA 3/7 – Treinamento CNN-LSTM com Optuna (walk-forward)"

for ATIVO in "${ATIVOS[@]}"; do
    INICIO_ATIVO=$(date +%s)
    echo ""
    echo "── ${ATIVO} ──"
    rodar src/train.py \
        --ativo "$ATIVO" \
        --modelo "$MODELO" \
        --epochs "$EPOCHS" \
        --optuna \
        --n-trials "$OPTUNA_TRIALS" \
        $GPU_FLAG

    verde "  [OK] ${ATIVO} concluído em $(tempo_decorrido $INICIO_ATIVO)"
done

# ============================================================================
# ETAPA 4 – ANÁLISE DOS MODELOS SALVOS
# ============================================================================
etapa "ETAPA 4/7 – Análise detalhada dos modelos salvos"

for ATIVO in "${ATIVOS[@]}"; do
    echo "── ${ATIVO} ──"
    rodar src/scripts/analisar_modelos_salvos.py \
        --ativo "$ATIVO" \
        --modelo "$MODELO"
done

# ============================================================================
# ETAPA 5 – BACKTESTS (todos os folds, ambas estratégias)
# ============================================================================
etapa "ETAPA 5/7 – Backtests com custos reais (long-only + long-short)"

for ATIVO in "${ATIVOS[@]}"; do
    echo ""
    echo "── ${ATIVO} ──"
    for FOLD in $(seq 1 $N_FOLDS); do
        for ESTRATEGIA in long_only long_short; do
            echo "  Fold ${FOLD} – ${ESTRATEGIA}"
            rodar src/scripts/rodar_backtest.py \
                --ativo "$ATIVO" \
                --fold "$FOLD" \
                --estrategia "$ESTRATEGIA"
        done
    done
    # Sensibilidade no fold 1 (representativo)
    echo "  Sensibilidade a custos (fold 1)"
    rodar src/scripts/rodar_backtest.py \
        --ativo "$ATIVO" \
        --fold 1 \
        --estrategia long_short \
        --sensibilidade
done

# ============================================================================
# ETAPA 6 – TESTES ESTATÍSTICOS E COMPARAÇÃO
# ============================================================================
etapa "ETAPA 6/7 – Testes estatísticos e comparação de modelos"

echo "── Comparativo CNN-LSTM vs Baselines ──"
rodar src/scripts/comparar_modelos.py

echo ""
echo "── Testes de Diebold-Mariano ──"
rodar src/scripts/rodar_testes_estatisticos.py \
    --todos \
    --regimes \
    --brier

echo ""
echo "── Tabelas e gráficos DM ──"
rodar src/scripts/gerar_tabelas_graficos_dm.py --grafico

# ============================================================================
# ETAPA 7 – SENSIBILIDADE DO WALK-FORWARD
# ============================================================================
etapa "ETAPA 7/7 – Análise de sensibilidade do walk-forward"

for ATIVO in "${ATIVOS[@]}"; do
    echo "── ${ATIVO} ──"
    rodar src/tests/testar_sensibilidade_walkforward.py --ativo "$ATIVO"
done

# ============================================================================
# RESUMO FINAL
# ============================================================================
etapa "PIPELINE CONCLUÍDO"

echo "Tempo total: $(tempo_decorrido $INICIO_GLOBAL)"
echo ""
echo "Artefatos gerados:"
echo "  data/processed/          Métricas, comparativos, testes DM"
echo "  data/backtest/           Backtests por fold e estratégia"
echo "  data/visualizacoes/      Gráficos de features por ativo/ano"
echo "  models/                  Modelos .keras por ativo/fold"
echo "  logs/training_history/   Histórico de treinamento por fold"
echo "  logs/pipeline/           Log desta execução"
echo ""
echo "Backup dos dados anteriores em: ${BACKUP_DIR}/"
echo ""

# Listar CSVs gerados
echo "CSVs em data/processed/:"
find data/processed -name '*.csv' -o -name '*.json' | sort | while read -r f; do
    echo "  $f"
done

echo ""
echo "CSVs em data/backtest/:"
find data/backtest -name '*.csv' | sort | while read -r f; do
    echo "  $f"
done

echo ""
verde "Pipeline completo. Pronto para usar no TCC."
