#!/bin/bash
# Script para treinar apenas os folds 4 e 5 do modelo CNN-LSTM
# Uso: ./treinar_folds_4_5.sh

echo "======================================================================"
echo "TREINAMENTO DOS FOLDS 4 E 5 - CNN-LSTM"
echo "======================================================================"
echo ""
echo "Configuração:"
echo "  - Ativo: VALE3"
echo "  - Modelo: CNN-LSTM"
echo "  - Folds: 4 e 5"
echo "  - Trials: 50"
echo "  - Epochs: 150"
echo ""
echo "======================================================================"
echo ""

# Criar diretórios necessários
mkdir -p logs/training_history/VALE3/cnn_lstm
mkdir -p models/VALE3/cnn_lstm

# Criar arquivo de log com timestamp
LOG_FILE="logs/treinamento_folds_4_5_$(date +%Y%m%d_%H%M%S).log"
echo "Salvando logs em: ${LOG_FILE}"
echo ""

# Iniciar treinamento
echo "Iniciando treinamento..."
echo ""

uv run python src/train.py \
    --ativo VALE3 \
    --modelo cnn_lstm \
    --optuna \
    --n-trials 50 \
    --epochs 150 \
    --folds 4,5 \
    --gpu \
    2>&1 | tee "${LOG_FILE}"

# Capturar exit code
EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "TREINAMENTO FINALIZADO"
echo "======================================================================"
echo ""
echo "Exit code: ${EXIT_CODE}"
echo "Log salvo em: ${LOG_FILE}"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Treinamento concluído com sucesso!"
    echo ""
    echo "Resultados:"
    echo "  - Métricas: data/processed/VALE3_cnn_lstm_walkforward.csv"
    echo "  - Modelos: models/VALE3/cnn_lstm/"
    echo "  - Histórico epochs: logs/training_history/VALE3/cnn_lstm/"
    echo ""
    echo "Verificar modelos salvos:"
    ls -lh models/VALE3/cnn_lstm/
else
    echo "❌ Treinamento falhou com erro ${EXIT_CODE}"
    echo ""
fi

echo "======================================================================"
