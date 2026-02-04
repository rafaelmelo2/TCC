"""Teste de Diebold-Mariano para comparação de acurácia preditiva entre dois modelos.

Conforme TCC Seção 4.5.2 (Testes de Robustez e Significância): comparação da série
de perdas/erros do modelo proposto contra baselines por meio do teste de
Diebold-Mariano (DIEBOLD; MARIANO, 1995), avaliando significância das diferenças
de acurácia direcional (e opcionalmente Brier).

Referências:
- DIEBOLD, F. X.; MARIANO, R. S. Comparing predictive accuracy. Journal of
  Business & Economic Statistics, 1995.
- HARVEY, D. I.; LEYBOURNE, S. J.; NEWBOLD, P. Testing the equality of
  prediction mean squared errors. International Journal of Forecasting, 1997.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Tuple, Optional


def perda_direcional(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Série de perda 0/1 para acurácia direcional (só observações não neutras).

    Perda = 1 se previsão direcional errada, 0 se correta. Neutros (y_true == 0)
    são excluídos, pois não há direção a prever.

    Parâmetros
    ----------
    y_true : np.ndarray
        Rótulos reais (-1, 0, 1).
    y_pred : np.ndarray
        Previsões (-1 ou 1; ou probabilidade em [0,1] para threshold 0.5).

    Retornos
    --------
    np.ndarray
        Vetor de perdas 0/1, comprimento = número de observações com y_true != 0.
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return np.array([], dtype=float)
    y_true_bin = (y_true[mask] > 0).astype(int)
    if np.issubdtype(y_pred.dtype, np.floating) and y_pred.max() <= 1 and y_pred.min() >= 0:
        y_pred_bin = (y_pred[mask] > 0.5).astype(int)
    else:
        y_pred_bin = (y_pred[mask] > 0).astype(int)
    return (y_true_bin != y_pred_bin).astype(float)


def perda_brier(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Série de perda Brier (y_prob - y_bin)^2 para observações não neutras.

    y_bin = 1 se y_true > 0, 0 se y_true < 0. Neutros (y_true == 0) são excluídos.

    Parâmetros
    ----------
    y_true : np.ndarray
        Rótulos reais (-1, 0, 1).
    y_prob : np.ndarray
        Probabilidades previstas em [0, 1] (ou sinais 0/1 para baselines).

    Retornos
    --------
    np.ndarray
        Vetor de perdas Brier, comprimento = número de observações com y_true != 0.
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return np.array([], dtype=float)
    y_bin = (y_true[mask] > 0).astype(float)
    p = np.clip(np.asarray(y_prob[mask], dtype=float).ravel(), 0.0, 1.0)
    return ((p - y_bin) ** 2).astype(float)


def perdas_direcionais_alinhadas(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perdas direcionais de dois modelos alinhadas aos mesmos índices (não neutros).

    Útil para aplicar o teste de Diebold-Mariano: ambos os modelos devem ser
    avaliados nas mesmas observações.

    Parâmetros
    ----------
    y_true : np.ndarray
        Rótulos reais (mesmo tamanho que as previsões).
    y_pred_a : np.ndarray
        Previsões do modelo A.
    y_pred_b : np.ndarray
        Previsões do modelo B.

    Retornos
    --------
    loss_a, loss_b : np.ndarray
        Séries de perda 0/1 para A e B, apenas em pontos não neutros.
    """
    mask = y_true != 0
    loss_a = perda_direcional(y_true, y_pred_a)
    loss_b = perda_direcional(y_true, y_pred_b)
    return loss_a, loss_b


def _variancia_bartlett(d: np.ndarray, h: int) -> float:
    """Estimativa da variância de longa duração de d com kernel de Bartlett (lag h)."""
    n = len(d)
    d_centered = d - np.mean(d)
    gamma_0 = np.var(d_centered, ddof=1) if n > 1 else 0.0
    if h <= 1 or n < 3:
        return max(gamma_0, 1e-20)
    v = gamma_0
    for j in range(1, min(h, n)):
        w = 1 - j / h
        gamma_j = np.mean(d_centered[:-j] * d_centered[j:]) if n > j else 0
        v += 2 * w * gamma_j
    return max(v, 1e-20)


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    h: int = 1,
    alternativa: str = "two_sided",
) -> Tuple[float, float]:
    """Teste de Diebold-Mariano para igualdade de acurácia preditiva.

    H0: E[loss_A] = E[loss_B]. A estatística DM compara a média da série
    de diferença de perdas d_t = loss_a_t - loss_b_t, com variância estimada
    por kernel de Bartlett para correção de autocorrelação (previsão h-passos).

    Parâmetros
    ----------
    loss_a : np.ndarray
        Série de perdas do modelo A (ex.: 0/1 por acurácia direcional).
    loss_b : np.ndarray
        Série de perdas do modelo B (mesmo tamanho e alinhamento que loss_a).
    h : int
        Horizonte de previsão; usado no kernel de Bartlett para variância HAC.
        h=1 para one-step (variância simples).
    alternativa : str
        "two_sided" (default), "less" (A melhor que B), "greater" (B melhor que A).

    Retornos
    --------
    dm_stat : float
        Estatística do teste (aproximação normal).
    p_value : float
        P-valor sob H0.
    """
    if len(loss_a) != len(loss_b):
        raise ValueError("loss_a e loss_b devem ter o mesmo tamanho")
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if n < 2:
        return np.nan, np.nan
    d_bar = np.mean(d)
    var_d = _variancia_bartlett(d, h)
    dm_stat = d_bar / np.sqrt(var_d / n)
    if alternativa == "two_sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    elif alternativa == "less":
        p_value = stats.norm.cdf(dm_stat)
    elif alternativa == "greater":
        p_value = 1 - stats.norm.cdf(dm_stat)
    else:
        raise ValueError("alternativa deve ser 'two_sided', 'less' ou 'greater'")
    return float(dm_stat), float(p_value)


def teste_paired_folds(
    acc_a: np.ndarray,
    acc_b: np.ndarray,
    teste: str = "wilcoxon",
) -> Tuple[float, float]:
    """Teste pareado por folds: compara acurácias (ou outras métricas) fold a fold.

    Útil quando só se dispõe de uma métrica por fold (ex.: acurácia direcional
    por fold) em vez de perdas por observação. Não substitui o Diebold-Mariano
    quando há séries de perdas disponíveis.

    Parâmetros
    ----------
    acc_a : np.ndarray
        Métrica por fold do modelo A (ex.: 5 folds).
    acc_b : np.ndarray
        Métrica por fold do modelo B (mesmo número de folds).
    teste : str
        "wilcoxon" (default) ou "ttest".

    Retornos
    --------
    stat : float
        Estatística do teste.
    p_value : float
        P-valor (bilateral).
    """
    if len(acc_a) != len(acc_b):
        raise ValueError("acc_a e acc_b devem ter o mesmo tamanho")
    if teste == "wilcoxon":
        stat, p_value = stats.wilcoxon(acc_a, acc_b, alternative="two-sided")
    elif teste == "ttest":
        stat, p_value = stats.ttest_rel(acc_a, acc_b)
    else:
        raise ValueError("teste deve ser 'wilcoxon' ou 'ttest'")
    return float(stat), float(p_value)


def resumo_dm(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    nome_a: str = "Modelo A",
    nome_b: str = "Modelo B",
    h: int = 1,
) -> dict:
    """Calcula diferença de perdas, estatística DM e p-valor; retorna dicionário resumo."""
    dm_stat, p_value = diebold_mariano(loss_a, loss_b, h=h)
    mean_a = np.mean(loss_a)
    mean_b = np.mean(loss_b)
    n = len(loss_a)
    return {
        "modelo_a": nome_a,
        "modelo_b": nome_b,
        "n_obs": n,
        "perda_media_a": mean_a,
        "perda_media_b": mean_b,
        "diferenca_perda": mean_a - mean_b,
        "dm_statistic": dm_stat,
        "dm_pvalue": p_value,
    }


def segmentar_por_volatilidade(
    volatility: np.ndarray,
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    percentil: float = 50.0,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Segmenta perdas por regime de volatilidade (baixa vs alta).

    Conforme TCC 4.5.2: segmentar resultados por regimes de volatilidade
    (calmaria vs. choques) para verificar estabilidade.

    Parâmetros
    ----------
    volatility : np.ndarray
        Volatilidade por observação (mesmo tamanho e ordem que loss_a/loss_b).
    loss_a : np.ndarray
        Série de perdas do modelo A.
    loss_b : np.ndarray
        Série de perdas do modelo B.
    percentil : float
        Percentil para corte (50 = mediana): baixa <= percentil, alta > percentil.

    Retornos
    --------
    (loss_a_baixa, loss_b_baixa), (loss_a_alta, loss_b_alta)
        Perdas em regime de baixa e alta volatilidade.
    """
    if len(volatility) != len(loss_a) or len(loss_a) != len(loss_b):
        raise ValueError("volatility, loss_a e loss_b devem ter o mesmo tamanho")
    limiar = np.nanpercentile(volatility, percentil)
    mask_baixa = np.isfinite(volatility) & (volatility <= limiar)
    mask_alta = np.isfinite(volatility) & (volatility > limiar)
    baixa = (loss_a[mask_baixa], loss_b[mask_baixa])
    alta = (loss_a[mask_alta], loss_b[mask_alta])
    return baixa, alta
