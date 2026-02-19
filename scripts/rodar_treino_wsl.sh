#!/bin/bash
# Rodar treinamento no WSL (execute a partir da raiz do repositório ou do diretório scripts/)

cd "$(dirname "$0")/.."
source .venv_wsl/bin/activate 2>/dev/null || true

# Adicionar CUDA ao PATH se necessário
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

# Rodar treinamento
python src/train.py --ativo VALE3 --modelo cnn_lstm --optuna --n-trials 30
