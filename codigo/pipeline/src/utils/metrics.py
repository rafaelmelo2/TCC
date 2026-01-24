"""Cálculo de métricas de avaliação."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, brier_score_loss, log_loss,
    precision_recall_curve, auc
)

from ..config import THRESHOLD_BANDA_MORTA


def calcular_acuracia_direcional(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0) -> float:
    """Calcula acurácia direcional (ignora apenas valores exatamente zero)."""
    # Ignora apenas valores exatamente zero (neutros reais)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    
    y_true_bin = (y_true[mask] > 0).astype(int)
    y_pred_bin = (y_pred[mask] > 0).astype(int)
    return accuracy_score(y_true_bin, y_pred_bin)


def calcular_metricas_preditivas(y_true: np.ndarray, y_pred: np.ndarray,
                                 y_prob: np.ndarray = None,
                                 threshold: float = 0.0) -> dict:
    """Calcula todas as métricas preditivas (ignora apenas valores exatamente zero)."""
    # Ignora apenas valores exatamente zero (neutros reais)
    mask = y_true != 0
    if mask.sum() == 0:
        return {}
    
    y_true_bin = (y_true[mask] > 0).astype(int)
    y_pred_bin = (y_pred[mask] > 0).astype(int)
    
    metricas = {
        'accuracy': accuracy_score(y_true_bin, y_pred_bin),
        'balanced_accuracy': balanced_accuracy_score(y_true_bin, y_pred_bin),
        'f1_score': f1_score(y_true_bin, y_pred_bin, zero_division=0),
        'mcc': matthews_corrcoef(y_true_bin, y_pred_bin)
    }
    
    if y_prob is not None:
        y_prob_filtered = y_prob[mask]
        if y_prob_filtered.min() >= 0 and y_prob_filtered.max() <= 1:
            metricas['brier_score'] = brier_score_loss(y_true_bin, y_prob_filtered)
            metricas['log_loss'] = log_loss(y_true_bin, y_prob_filtered)
            try:
                precision, recall, _ = precision_recall_curve(y_true_bin, y_prob_filtered)
                metricas['auc_pr'] = auc(recall, precision)
            except:
                metricas['auc_pr'] = 0.0
    
    return metricas


def calcular_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calcular_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))
