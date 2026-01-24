#!/bin/bash
# Rodar treinamento no WSL

cd /mnt/d/Rafael/TCC/codigo/pipeline
source .venv_wsl/bin/activate

# Adicionar CUDA ao PATH se necessário
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

# Rodar treinamento
python src/train.py --ativo VALE3 --modelo cnn_lstm --optuna --n-trials 30
