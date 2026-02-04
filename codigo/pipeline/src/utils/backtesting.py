"""
Backtesting com custos de transação e métricas financeiras.

Conforme TCC Seção 4.5.1: "Executamos backtests long-only e long/short
condicionados às probabilidades previstas e a limiares calibrados. Custos
fixos e proporcionais (corretagem, emolumentos) e slippage são descontados;
também reportamos turnover e sensibilidade a custos (análise de estresse),
em conformidade com as práticas e estruturas de custos divulgadas pela B3."

Métricas calculadas (TCC e literatura):
- Retorno líquido (após custos)
- Índice de Sharpe (SHARPE, 1994)
- Maximum Drawdown
- Profit Factor
- Turnover (frequência/volume de negociação)

Uso típico:
    from src.utils.backtesting import (
        CustosBacktest,
        run_backtest,
        run_backtest_sensibilidade_custos,
    )
    custos = CustosBacktest.from_config()
    resultado = run_backtest(
        returns_realized=retornos_periodo,
        signal=sinais,
        custos=custos,
        estrategia="long_short",
    )
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# -----------------------------------------------------------------------------
# Constantes e configuração
# -----------------------------------------------------------------------------

# Barras de 15 minutos: ~26 barras/dia (6h30), ~252 dias/ano
# Usado para annualizar Sharpe (retorno por unidade de risco anualizada).
BARRAS_POR_ANO = 26 * 252  # 6552


@dataclass
class CustosBacktest:
    """
    Parâmetros de custos de transação para backtest.

    Conforme TCC e regras B3: custos fixos (corretagem), proporcionais
    (emolumentos) e slippage. Valores em unidades compatíveis com o capital:
    - custo_fixo: R$ por operação (ex.: corretagem)
    - custo_proporcional: fração do valor negociado (ex.: 0.0003 = 0.03%)
    - slippage: fração do valor negociado (ex.: 0.0001 = 0.01%)
    - capital_inicial: R$ no início do backtest
    """

    custo_fixo: float = 0.0
    custo_proporcional: float = 0.0003
    slippage: float = 0.0005
    capital_inicial: float = 100_000.0

    @classmethod
    def from_config(cls) -> CustosBacktest:
        """Cria CustosBacktest a partir de src.config (CUSTO_*, CAPITAL_INICIAL)."""
        try:
            from ..config import (
                CAPITAL_INICIAL,
                CUSTO_CORRETAGEM,
                CUSTO_SLIPPAGE,
                CUSTO_TAXA_PROPORCIONAL,
            )
            return cls(
                custo_fixo=CUSTO_CORRETAGEM,
                custo_proporcional=CUSTO_TAXA_PROPORCIONAL,
                slippage=CUSTO_SLIPPAGE,
                capital_inicial=CAPITAL_INICIAL,
            )
        except ImportError:
            return cls()


# -----------------------------------------------------------------------------
# Conversão de sinal e posição
# -----------------------------------------------------------------------------

def _sinal_para_posicao(
    signal: np.ndarray,
    estrategia: Literal["long_only", "long_short"],
) -> np.ndarray:
    """
    Converte sinal de previsão em posição efetiva.

    - long_only: apenas compra (1) ou neutro (0); sinal -1 vira 0.
    - long_short: compra (1), venda a descoberto (-1) ou neutro (0).

    Parâmetros:
        signal: 1 (alta), -1 (baixa), 0 (neutro)
        estrategia: "long_only" ou "long_short"

    Retorna:
        position: 1, -1 ou 0, alinhado ao tamanho de signal
    """
    signal = np.asarray(signal, dtype=float)
    if estrategia == "long_only":
        return np.where(signal < 0, 0.0, signal)  # -1 -> 0
    return signal.copy()


def _custo_em_reais(
    capital: float,
    custos: CustosBacktest,
    n_operacoes: int = 1,
) -> float:
    """
    Custo total em R$ em uma mudança de posição.

    Considera: custo fixo por operação + proporcional e slippage sobre
    o valor exposto (capital * |posição|). Para uma operação, exposição = capital.

    Parâmetros:
        capital: capital no momento da operação (R$)
        custos: parâmetros de custo
        n_operacoes: número de "pernas" (ex.: 1 para 0->1, 2 para 1->-1)

    Retorna:
        Custo em R$
    """
    valor_exposto = capital
    custo_variavel = (custos.custo_proporcional + custos.slippage) * valor_exposto
    return n_operacoes * custos.custo_fixo + custo_variavel


# -----------------------------------------------------------------------------
# Motor do backtest
# -----------------------------------------------------------------------------

def run_backtest(
    returns_realized: np.ndarray,
    signal: np.ndarray,
    custos: CustosBacktest | None = None,
    estrategia: Literal["long_only", "long_short"] = "long_short",
    barras_por_ano: int = BARRAS_POR_ANO,
) -> dict:
    """
    Executa backtest barra a barra com custos e retorna métricas.

    A convenção temporal é: signal[i] é a decisão no início do período i;
    returns_realized[i] é o retorno logarítmico realizado no período i (o que
    se ganharia estando long). P&L do período: posição[i] * returns_realized[i]
    (para short, posição = -1).

    Custos são aplicados apenas quando a posição muda (entrada ou saída).
    Round-trip 1->-1 conta como duas operações (fechar long, abrir short).

    Parâmetros:
        returns_realized: retornos logarítmicos realizados por período (N,)
        signal: sinal de previsão por período: 1 (alta), -1 (baixa), 0 (neutro); (N,)
        custos: parâmetros de custo; se None, usa CustosBacktest.from_config()
        estrategia: "long_only" (apenas compra/neutro) ou "long_short"
        barras_por_ano: usado para annualizar o Sharpe

    Retorna:
        Dicionário com:
        - retorno_liquido: (capital_final - capital_inicial) / capital_inicial
        - sharpe_ratio: anualizado, (média dos retornos líquidos / desvio) * sqrt(barras_por_ano)
        - max_drawdown: maior queda percentual do pico ao vale da curva de capital
        - profit_factor: soma(retornos positivos) / |soma(retornos negativos)|
        - turnover: número de mudanças de posição / número de barras (frequência)
        - n_trades: número de operações (mudanças de posição, contando 1->-1 como 2)
        - custo_total_reais: total pago em custos (R$)
        - capital_final: capital ao final (R$)
        - equity_curve: array (N+1,) com capital no início de cada período
        - period_returns: retornos líquidos por período (após custos, em log para consistência)
    """
    returns_realized = np.asarray(returns_realized, dtype=float).flatten()
    signal = np.asarray(signal, dtype=float).flatten()
    n = len(returns_realized)
    if len(signal) != n:
        raise ValueError(
            f"returns_realized e signal devem ter o mesmo tamanho; obtidos {n} e {len(signal)}"
        )

    if custos is None:
        custos = CustosBacktest.from_config()

    posicao = _sinal_para_posicao(signal, estrategia)

    # Curva de capital e retornos líquidos por período
    capital = custos.capital_inicial
    equity_curve = np.empty(n + 1)
    equity_curve[0] = capital
    period_returns = np.zeros(n)  # retorno líquido por período (em termos simples para capital)
    custo_total = 0.0
    posicao_anterior = 0.0

    for i in range(n):
        # P&L bruto do período (em log: posição * retorno_log)
        pnl_log = posicao[i] * returns_realized[i]
        retorno_simples = np.exp(pnl_log) - 1.0
        capital_antes = capital
        capital = capital * (1.0 + retorno_simples)

        # Custos apenas quando a posição muda
        if posicao[i] != posicao_anterior:
            n_operacoes = 2 if (posicao_anterior != 0 and posicao[i] != 0) else 1
            custo_i = _custo_em_reais(capital_antes, custos, n_operacoes=n_operacoes)
            custo_total += custo_i
            capital -= custo_i
            # Não deixar capital negativo por custos
            capital = max(capital, 1.0)

        equity_curve[i + 1] = capital
        period_returns[i] = (capital - capital_antes) / capital_antes if capital_antes > 0 else 0.0
        posicao_anterior = posicao[i]

    # Métricas
    retorno_liquido = (capital - custos.capital_inicial) / custos.capital_inicial
    # Número de operações (pernas): 0->1 ou 1->0 = 1; 1->-1 = 2
    n_trades = 0
    p_ant = 0.0
    for i in range(n):
        p_atual = posicao[i]
        if p_ant != p_atual:
            n_trades += 2 if (p_ant != 0 and p_atual != 0) else 1
        p_ant = p_atual

    sharpe_ratio = _calcular_sharpe(period_returns, barras_por_ano)
    max_drawdown = _calcular_max_drawdown(equity_curve)
    profit_factor = _calcular_profit_factor(period_returns)
    turnover = _calcular_turnover(posicao, n)

    return {
        "retorno_liquido": retorno_liquido,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "turnover": turnover,
        "n_trades": n_trades,
        "custo_total_reais": custo_total,
        "capital_final": capital,
        "equity_curve": equity_curve,
        "period_returns": period_returns,
    }


def _calcular_sharpe(
    period_returns: np.ndarray,
    barras_por_ano: int,
) -> float:
    """Sharpe anualizado: média / desvio * sqrt(barras_por_ano). Retorna 0 se desvio = 0."""
    if len(period_returns) == 0:
        return 0.0
    media = np.mean(period_returns)
    std = np.std(period_returns)
    if std <= 0:
        return 0.0
    return (media / std) * np.sqrt(barras_por_ano)


def _calcular_max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown: maior queda percentual do pico ao vale."""
    if len(equity_curve) < 2:
        return 0.0
    pico = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - pico) / np.where(pico > 0, pico, 1.0)
    return float(np.min(drawdowns))  # negativo, ex: -0.05 = 5%


def _calcular_profit_factor(period_returns: np.ndarray) -> float:
    """Profit factor: soma(ganhos) / |soma(perdas)|. Retorna 0 se não houver perdas."""
    ganhos = np.sum(period_returns[period_returns > 0])
    perdas = np.sum(period_returns[period_returns < 0])
    if perdas >= 0 or perdas == 0:
        return 0.0 if ganhos == 0 else np.inf
    return ganhos / abs(perdas)


def _calcular_turnover(posicao: np.ndarray, n: int) -> float:
    """Turnover: número de mudanças de posição / número de barras."""
    if n == 0:
        return 0.0
    mudancas = np.sum(np.abs(np.diff(posicao, prepend=posicao[0] if len(posicao) else 0)) > 0)
    return mudancas / n


# -----------------------------------------------------------------------------
# Sensibilidade a custos (análise de estresse)
# -----------------------------------------------------------------------------

def run_backtest_sensibilidade_custos(
    returns_realized: np.ndarray,
    signal: np.ndarray,
    custos_base: CustosBacktest | None = None,
    estrategia: Literal["long_only", "long_short"] = "long_short",
    multiplicadores_custo: list[float] | None = None,
) -> list[dict]:
    """
    Roda o backtest para vários cenários de custo (análise de estresse).

    Útil para ver como retorno líquido, Sharpe e turnover variam quando
    custos aumentam (ex.: 1x, 1.5x, 2x os custos base), em linha com o TCC
    (sensibilidade a custos e turnover).

    Parâmetros:
        returns_realized: retornos log por período
        signal: sinal por período
        custos_base: custos de referência; se None, usa from_config()
        estrategia: "long_only" ou "long_short"
        multiplicadores_custo: fatores a aplicar ao custo (ex.: [0.5, 1.0, 1.5, 2.0])

    Retorna:
        Lista de dicionários, um por cenário, cada um com as chaves do
        resultado de run_backtest mais "multiplicador_custo" e "custo_total_reais".
    """
    if custos_base is None:
        custos_base = CustosBacktest.from_config()
    if multiplicadores_custo is None:
        multiplicadores_custo = [0.5, 1.0, 1.5, 2.0]

    resultados = []
    for mult in multiplicadores_custo:
        custos_cenario = CustosBacktest(
            custo_fixo=custos_base.custo_fixo * mult,
            custo_proporcional=custos_base.custo_proporcional * mult,
            slippage=custos_base.slippage * mult,
            capital_inicial=custos_base.capital_inicial,
        )
        r = run_backtest(
            returns_realized=returns_realized,
            signal=signal,
            custos=custos_cenario,
            estrategia=estrategia,
        )
        r["multiplicador_custo"] = mult
        resultados.append(r)
    return resultados


# -----------------------------------------------------------------------------
# Helpers para uso com dados do pipeline
# -----------------------------------------------------------------------------

def sinal_de_probabilidade(
    proba: np.ndarray,
    limiar_alta: float = 0.5,
    limiar_baixa: float = 0.5,
) -> np.ndarray:
    """
    Converte probabilidade prevista (P(alta)) em sinal discreto 1 / -1 / 0.

    Parâmetros:
        proba: probabilidade de alta (0 a 1), shape (N,)
        limiar_alta: acima deste valor -> sinal 1 (long)
        limiar_baixa: abaixo de (1 - limiar_baixa) -> sinal -1 (short).
                     Por simetria, usa-se normalmente limiar_baixa = 1 - limiar_alta.

    Retorna:
        signal: 1 (alta), -1 (baixa), 0 (neutro). Neutro quando
                limiar_baixa <= proba <= limiar_alta (ex.: 0.45--0.55).
    """
    proba = np.asarray(proba, dtype=float).flatten()
    signal = np.zeros_like(proba)
    signal[proba > limiar_alta] = 1.0
    signal[proba < (1.0 - limiar_baixa)] = -1.0
    return signal


def retornos_e_sinal_para_backtest(
    returns: np.ndarray,
    signal: np.ndarray,
    indice_test_start: int,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Alinha retornos realizados e sinal para o backtest a partir de índices do walk-forward.

    O modelo prevê a direção do retorno na próxima barra. O target na amostra j é a direção
    do retorno na barra (indice_test_start + n_steps + j); esse retorno ocorre na barra
    seguinte: indice_test_start + n_steps + j + 1. Então returns_realized[j] = returns
    [indice_test_start + n_steps + j + 1].

    Parâmetros:
        returns: série completa de retornos logarítmicos
        signal: sinal previsto para cada amostra de teste (mesmo tamanho que o teste)
        indice_test_start: primeiro índice do bloco de teste no array returns
        n_steps: janela temporal (número de barras na sequência)

    Retorna:
        returns_realized: retornos realizados alinhados ao signal (len = len(signal))
        signal: sinal truncado se retornos forem insuficientes (para compatibilidade)
    """
    returns = np.asarray(returns).flatten()
    signal = np.asarray(signal).flatten()
    n = len(signal)
    # Retorno realizado na barra seguinte ao "target" (próxima barra após a janela)
    inicio = indice_test_start + n_steps + 1
    n_disponivel = len(returns) - inicio
    if n_disponivel < n:
        warnings.warn(
            f"retornos_e_sinal_para_backtest: retornos disponíveis ({n_disponivel}) < "
            f"tamanho do sinal ({n}); será usado apenas o disponível.",
            UserWarning,
        )
        n = max(0, n_disponivel)
        signal = signal[:n]
    if n == 0:
        return np.array([]), np.array([])
    returns_realized = returns[inicio : inicio + n].copy()
    return returns_realized, signal


__all__ = [
    "CustosBacktest",
    "run_backtest",
    "run_backtest_sensibilidade_custos",
    "sinal_de_probabilidade",
    "retornos_e_sinal_para_backtest",
    "BARRAS_POR_ANO",
]
