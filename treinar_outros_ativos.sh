#!/bin/bash
# Script para treinar CNN-LSTM em PETR4 e ITUB4
# Uso: ./treinar_outros_ativos.sh

echo "======================================================================"
echo "TREINAMENTO CNN-LSTM - PETR4 E ITUB4"
echo "======================================================================"
echo ""
echo "Este script treinará os modelos em:"
echo "  - PETR4 (Petrobras)"
echo "  - ITUB4 (Itaú)"
echo ""
echo "Configuração:"
echo "  - Modelo: CNN-LSTM"
echo "  - Trials: 50"
echo "  - Epochs: 150"
echo "  - Folds: Todos (1-5)"
echo ""
echo "======================================================================"
echo ""

# Criar diretórios necessários
mkdir -p logs/training_history/{PETR4,ITUB4}/cnn_lstm
mkdir -p models/{PETR4,ITUB4}/cnn_lstm

# Função para treinar um ativo
treinar_ativo() {
    local ATIVO=$1
    local LOG_FILE="logs/treinamento_${ATIVO}_$(date +%Y%m%d_%H%M%S).log"
    
    echo ""
    echo "======================================================================"
    echo "TREINANDO ${ATIVO}"
    echo "======================================================================"
    echo "Log: ${LOG_FILE}"
    echo ""
    
    uv run python src/train.py \
        --ativo ${ATIVO} \
        --modelo cnn_lstm \
        --optuna \
        --n-trials 50 \
        --epochs 150 \
        --gpu \
        2>&1 | tee "${LOG_FILE}"
    
    local EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ ${ATIVO} concluído com sucesso!"
        echo "   Modelos: models/${ATIVO}/cnn_lstm/"
        echo "   Métricas: data/processed/${ATIVO}_cnn_lstm_walkforward.csv"
    else
        echo ""
        echo "❌ ${ATIVO} falhou com erro ${EXIT_CODE}"
    fi
    
    return $EXIT_CODE
}

# Treinar PETR4
echo "[1/2] Iniciando treinamento de PETR4..."
treinar_ativo PETR4
PETR4_EXIT=$?

# Treinar ITUB4
echo ""
echo "[2/2] Iniciando treinamento de ITUB4..."
treinar_ativo ITUB4
ITUB4_EXIT=$?

# Resumo final
echo ""
echo "======================================================================"
echo "RESUMO FINAL"
echo "======================================================================"
echo ""

if [ $PETR4_EXIT -eq 0 ]; then
    echo "✅ PETR4: Concluído"
    cat data/processed/PETR4_cnn_lstm_walkforward.csv 2>/dev/null | tail -n +2 | awk -F',' '{printf "   Fold %s: %.2f%%\n", $1, $2*100}'
else
    echo "❌ PETR4: Falhou"
fi

echo ""

if [ $ITUB4_EXIT -eq 0 ]; then
    echo "✅ ITUB4: Concluído"
    cat data/processed/ITUB4_cnn_lstm_walkforward.csv 2>/dev/null | tail -n +2 | awk -F',' '{printf "   Fold %s: %.2f%%\n", $1, $2*100}'
else
    echo "❌ ITUB4: Falhou"
fi

echo ""
echo "======================================================================"
echo "COMPARAÇÃO ENTRE ATIVOS"
echo "======================================================================"
echo ""

# Comparar resultados se todos completaram
if [ $PETR4_EXIT -eq 0 ] && [ $ITUB4_EXIT -eq 0 ]; then
    echo "Acurácia Média por Ativo:"
    echo ""
    
    for ATIVO in VALE3 PETR4 ITUB4; do
        if [ -f "data/processed/${ATIVO}_cnn_lstm_walkforward.csv" ]; then
            MEAN=$(awk -F',' 'NR>1 {sum+=$2; count++} END {if(count>0) printf "%.2f", (sum/count)*100; else printf "N/A"}' data/processed/${ATIVO}_cnn_lstm_walkforward.csv)
            echo "  ${ATIVO}: ${MEAN}%"
        fi
    done
fi

echo ""
echo "======================================================================"
