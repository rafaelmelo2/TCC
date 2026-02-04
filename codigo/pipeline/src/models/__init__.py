"""Modelos de machine learning."""

from .baselines import BaseBaseline, NaiveBaseline, DriftBaseline, ARIMABaseline
from .prophet_model import ProphetBaseline
# lstm e cnn_lstm carregados sob demanda para evitar import circular
# (optuna_optimizer -> lstm_model quando models é importado logo no início)

__all__ = [
    'BaseBaseline', 'NaiveBaseline', 'DriftBaseline', 'ARIMABaseline', 'ProphetBaseline',
    'criar_modelo_lstm', 'criar_modelo_cnn_lstm'
]


def __getattr__(name: str):
    if name == 'criar_modelo_lstm':
        from .lstm_model import criar_modelo_lstm
        return criar_modelo_lstm
    if name == 'criar_modelo_cnn_lstm':
        from .cnn_lstm_model import criar_modelo_cnn_lstm
        return criar_modelo_cnn_lstm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
