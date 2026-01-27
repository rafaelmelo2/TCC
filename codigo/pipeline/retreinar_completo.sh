#!/bin/bash
set -e

echo "=========================================="
echo "RETREINAMENTO COMPLETO - TCC"
echo "Data: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

TRIALS=20
EPOCHS=100

# Lista de ativos
ATIVOS=("PETR4" "VALE3" "ITUB4")

echo "Configuração:"
echo "  Trials Optuna: ${TRIALS}"
echo "  Épocas: ${EPOCHS}"
echo "  Focal Loss: ATIVO (gamma=5.0, alpha=0.5)"
echo "  Modelo: CNN-LSTM"
echo "  Folds: 5 (walk-forward)"
echo ""

# Contador de progresso
TOTAL=${#ATIVOS[@]}
ATUAL=0

for ATIVO in "${ATIVOS[@]}"; do
    ATUAL=$((ATUAL + 1))
    
    echo ""
    echo "=========================================="
    echo "[${ATUAL}/${TOTAL}] Treinando ${ATIVO}..."
    echo "=========================================="
    echo ""
    
    # Timestamp de início
    START_TIME=$(date +%s)
    
    # Executar treinamento
    uv run python src/train.py \
        --ativo ${ATIVO} \
        --modelo cnn_lstm \
        --optuna \
        --n-trials ${TRIALS} \
        --epochs ${EPOCHS} \
        --focal-loss || {
            echo ""
            echo "⚠️  ERRO ao treinar ${ATIVO}!"
            echo "Continuando para próximo ativo..."
            continue
        }
    
    # Calcular tempo decorrido
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))
    
    echo ""
    echo "✅ ${ATIVO} concluído em ${MINUTES}m ${SECONDS}s"
    echo ""
done

echo ""
echo "=========================================="
echo "TREINAMENTO CONCLUÍDO!"
echo "Data: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""
echo "📊 Resultados salvos em:"
echo "  - data/processed/{ATIVO}_cnn_lstm_walkforward.csv"
echo "  - models/{ATIVO}/cnn_lstm/fold_*_checkpoint.keras"
echo "  - logs/training_history/{ATIVO}/cnn_lstm/"
echo ""
echo "📝 Próximos passos:"
echo "  1. Analisar resultados:"
echo "     uv run python src/scripts/analisar_modelos_salvos.py"
echo ""
echo "  2. Ver previsões:"
echo "     cat data/processed/PETR4_cnn_lstm_walkforward.csv"
echo ""
echo "  3. Comparar com baselines:"
echo "     uv run python src/scripts/comparar_modelos.py"
echo ""
echo "✅ Tudo pronto para análise final do TCC!"
