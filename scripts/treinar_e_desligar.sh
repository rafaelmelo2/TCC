#!/bin/bash
# Script para treinar e desligar automaticamente
# Uso: ./treinar_e_desligar.sh [horas_ate_desligar]

# Parâmetros
HORAS_ATE_DESLIGAR=${1:-3}  # Padrão: 3 horas
MINUTOS_ATE_DESLIGAR=$((HORAS_ATE_DESLIGAR * 60))

echo "======================================================================"
echo "TREINAMENTO COM DESLIGAMENTO AUTOMÁTICO"
echo "======================================================================"
echo ""
echo "Configuração:"
echo "  - Ativo: VALE3"
echo "  - Modelo: CNN-LSTM"
echo "  - Trials: 50"
echo "  - Epochs: 150"
echo "  - Desligamento em: ${HORAS_ATE_DESLIGAR} horas"
echo ""
echo "======================================================================"
echo ""

# Criar diretórios necessários
mkdir -p logs/training_history/VALE3/cnn_lstm
mkdir -p models/VALE3/cnn_lstm

# Agendar desligamento
echo "[1/3] Agendando desligamento em ${HORAS_ATE_DESLIGAR} horas..."
sudo shutdown -h +${MINUTOS_ATE_DESLIGAR}
echo "[OK] Sistema será desligado em ${HORAS_ATE_DESLIGAR} horas"
echo ""

# Mostrar quando vai desligar
echo "Desligamento agendado para: $(date -d "+${HORAS_ATE_DESLIGAR} hours" '+%H:%M:%S')"
echo ""

# Criar arquivo de log com timestamp
LOG_FILE="logs/treinamento_$(date +%Y%m%d_%H%M%S).log"
echo "[2/3] Salvando logs em: ${LOG_FILE}"
echo ""

# Iniciar treinamento
echo "[3/3] Iniciando treinamento..."
echo ""
echo "======================================================================"
echo ""

uv run python src/train.py \
    --ativo VALE3 \
    --modelo cnn_lstm \
    --optuna \
    --n-trials 50 \
    --epochs 150 \
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
else
    echo "❌ Treinamento falhou com erro ${EXIT_CODE}"
    echo ""
fi

echo "======================================================================"
echo ""
echo "Sistema será desligado em alguns minutos..."
echo "Para cancelar: sudo shutdown -c"
echo ""
echo "======================================================================"
