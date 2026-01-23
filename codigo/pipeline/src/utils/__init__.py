"""
Módulo de utilitários.

Inclui métricas de avaliação e outras funções auxiliares.
"""

from .metrics import (
    calcular_acuracia_direcional,
    calcular_metricas_preditivas,
    calcular_metricas_completas,
    calcular_rmse,
    calcular_mae
)

__all__ = [
    'calcular_acuracia_direcional',
    'calcular_metricas_preditivas',
    'calcular_metricas_completas',
    'calcular_rmse',
    'calcular_mae'
]
