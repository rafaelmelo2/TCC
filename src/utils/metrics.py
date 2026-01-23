"""
Módulo para cálculo de métricas de avaliação.

Conforme metodologia do TCC (Seção 4.5 - Métricas de Avaliação).
Calcula métricas preditivas e de trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, brier_score_loss, log_loss,
    precision_recall_curve, auc
)

try:
    from ..config import THRESHOLD_BANDA_MORTA
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import THRESHOLD_BANDA_MORTA


def calcular_acuracia_direcional(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = THRESHOLD_BANDA_MORTA
) -> float:
    """
    Calcula acurácia direcional com banda morta.
    
    Ignora movimentos neutros (|retorno| < threshold).
    
    Parâmetros:
        y_true: Valores reais (direção: 1, -1, 0)
        y_pred: Valores previstos (direção: 1, -1, 0)
        threshold: Threshold da banda morta
        
    Retorna:
        Acurácia direcional (0-1)
    """
    # Filtrar apenas movimentos não-neutros
    mask = np.abs(y_true) > threshold
    if mask.sum() == 0:
        return 0.0
    
    y_true_filtered = (y_true[mask] > 0).astype(int)
    y_pred_filtered = (y_pred[mask] > 0).astype(int)
    
    return accuracy_score(y_true_filtered, y_pred_filtered)


def calcular_metricas_preditivas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = THRESHOLD_BANDA_MORTA
) -> Dict[str, float]:
    """
    Calcula todas as métricas preditivas.
    
    Conforme metodologia do TCC (Seção 4.5).
    
    Parâmetros:
        y_true: Valores reais (direção: 1, -1, 0)
        y_pred: Valores previstos (direção: 1, -1, 0)
        y_prob: Probabilidades previstas (opcional, para métricas probabilísticas)
        threshold: Threshold da banda morta
        
    Retorna:
        Dicionário com todas as métricas
    """
    # Filtrar banda morta
    mask = np.abs(y_true) > threshold
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
    
    # Métricas probabilísticas (se y_prob fornecido)
    if y_prob is not None:
        y_prob_filtered = y_prob[mask]
        
        # Converter probabilidades para binário se necessário
        if y_prob_filtered.min() >= 0 and y_prob_filtered.max() <= 1:
            # Já são probabilidades
            metricas['brier_score'] = brier_score_loss(y_true_bin, y_prob_filtered)
            metricas['log_loss'] = log_loss(y_true_bin, y_prob_filtered)
            
            # AUC-PR
            try:
                precision, recall, _ = precision_recall_curve(y_true_bin, y_prob_filtered)
                metricas['auc_pr'] = auc(recall, precision)
            except:
                metricas['auc_pr'] = 0.0
    
    return metricas


def calcular_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula Root Mean Squared Error.
    
    Parâmetros:
        y_true: Valores reais
        y_pred: Valores previstos
        
    Retorna:
        RMSE
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calcular_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula Mean Absolute Error.
    
    Parâmetros:
        y_true: Valores reais
        y_pred: Valores previstos
        
    Retorna:
        MAE
    """
    return np.mean(np.abs(y_true - y_pred))


def calcular_metricas_completas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    y_true_returns: Optional[np.ndarray] = None,
    y_pred_returns: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calcula todas as métricas (preditivas + regressão).
    
    Parâmetros:
        y_true: Valores reais (direção)
        y_pred: Valores previstos (direção)
        y_prob: Probabilidades previstas (opcional)
        y_true_returns: Retornos reais (opcional, para RMSE/MAE)
        y_pred_returns: Retornos previstos (opcional, para RMSE/MAE)
        
    Retorna:
        Dicionário com todas as métricas
    """
    metricas = calcular_metricas_preditivas(y_true, y_pred, y_prob)
    
    # Métricas de regressão (se retornos fornecidos)
    if y_true_returns is not None and y_pred_returns is not None:
        metricas['rmse'] = calcular_rmse(y_true_returns, y_pred_returns)
        metricas['mae'] = calcular_mae(y_true_returns, y_pred_returns)
    
    return metricas


if __name__ == '__main__':
    """
    Teste básico do módulo.
    """
    try:
        # Dados de exemplo
        y_true = np.array([1, 1, -1, -1, 0, 1, -1, 0, 1, -1])
        y_pred = np.array([1, -1, -1, 1, 0, 1, 1, 0, 1, -1])
        y_prob = np.array([0.8, 0.3, 0.2, 0.6, 0.5, 0.9, 0.4, 0.5, 0.7, 0.1])
        
        print("=" * 70)
        print("TESTE DE MÉTRICAS")
        print("=" * 70)
        
        metricas = calcular_metricas_completas(y_true, y_pred, y_prob)
        
        print("\nMétricas calculadas:")
        for nome, valor in metricas.items():
            print(f"  {nome}: {valor:.4f}")
        
        print("\n" + "=" * 70)
    except ImportError as e:
        print(f"[!] Dependências não instaladas: {e}")
        print("    Instale com: pip install scikit-learn")
