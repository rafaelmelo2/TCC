"""
Módulo de modelos de machine learning para o pipeline.
"""

from .baselines import (
    BaseBaseline,
    NaiveBaseline,
    DriftBaseline,
    ARIMABaseline,
    criar_baseline
)

__all__ = [
    'BaseBaseline',
    'NaiveBaseline',
    'DriftBaseline',
    'ARIMABaseline',
    'criar_baseline'
]
