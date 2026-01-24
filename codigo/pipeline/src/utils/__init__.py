"""Utilitários: métricas e validação."""

from .metrics import calcular_acuracia_direcional, calcular_metricas_preditivas, calcular_rmse, calcular_mae
from .validation import WalkForwardValidator, FoldInfo
from .optuna_optimizer import (
    otimizar_hiperparametros,
    criar_espaco_busca_lstm,
    criar_espaco_busca_cnn_lstm
)

__all__ = [
    'calcular_acuracia_direcional', 'calcular_metricas_preditivas',
    'calcular_rmse', 'calcular_mae', 'WalkForwardValidator', 'FoldInfo',
    'otimizar_hiperparametros', 'criar_espaco_busca_lstm', 'criar_espaco_busca_cnn_lstm'
]
