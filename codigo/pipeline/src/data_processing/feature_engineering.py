"""Engenharia de features (indicadores técnicos)."""

import pandas as pd
import numpy as np

from ..config import (
    PERIODOS_EMA, PERIODOS_RSI, PERIODO_BOLLINGER, DESVIOS_BOLLINGER,
    PERIODO_VOLATILIDADE, THRESHOLD_BANDA_MORTA
)


def calcular_retornos_logaritmicos(df: pd.DataFrame, coluna_preco: str = 'fechamento') -> pd.Series:
    """Calcula retornos logarítmicos: rt = ln(Pt) - ln(Pt-1)."""
    return np.log(df[coluna_preco] / df[coluna_preco].shift(1))


def calcular_ema(df: pd.DataFrame, periodo: int, coluna_preco: str = 'fechamento') -> pd.Series:
    """Calcula Média Móvel Exponencial (EMA)."""
    return df[coluna_preco].ewm(span=periodo, adjust=False).mean()


def calcular_rsi(df: pd.DataFrame, periodo: int = 14, coluna_preco: str = 'fechamento') -> pd.Series:
    """Calcula Relative Strength Index (RSI)."""
    delta = df[coluna_preco].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=periodo, adjust=False).mean()
    avg_loss = loss.ewm(span=periodo, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calcular_bandas_bollinger(df: pd.DataFrame, periodo: int = PERIODO_BOLLINGER,
                               num_desvios: float = DESVIOS_BOLLINGER,
                               coluna_preco: str = 'fechamento') -> pd.DataFrame:
    """Calcula Bandas de Bollinger."""
    sma = df[coluna_preco].rolling(window=periodo).mean()
    std = df[coluna_preco].rolling(window=periodo).std()
    bb_upper = sma + (std * num_desvios)
    bb_lower = sma - (std * num_desvios)
    bb_width = (bb_upper - bb_lower) / sma
    bb_position = (df[coluna_preco] - bb_lower) / (bb_upper - bb_lower)
    
    return pd.DataFrame({
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'bb_middle': sma,
        'bb_width': bb_width,
        'bb_position': bb_position
    })


def calcular_volatilidade(df: pd.DataFrame, periodo: int = PERIODO_VOLATILIDADE,
                          coluna_retornos: str = 'returns') -> pd.Series:
    """Calcula volatilidade realizada (desvio-padrão dos retornos)."""
    return df[coluna_retornos].rolling(window=periodo).std()


def criar_target_com_banda_morta(df: pd.DataFrame, coluna_retornos: str = 'returns',
                                  threshold: float = 0.0) -> pd.Series:
    """Cria target: 1 (alta), -1 (baixa), 0 (neutro apenas se retorno == 0)."""
    next_return = df[coluna_retornos].shift(-1)
    target = pd.Series(0, index=df.index, dtype=int)
    target.loc[next_return > threshold] = 1
    target.loc[next_return < -threshold] = -1
    # Apenas valores exatamente zero ficam como 0 (neutro)
    return target


def criar_features(df: pd.DataFrame, incluir_retornos: bool = True, incluir_ema: bool = True,
                   incluir_rsi: bool = True, incluir_bollinger: bool = True,
                   incluir_volatilidade: bool = True, incluir_target: bool = True,
                   verbose: bool = True) -> pd.DataFrame:
    """Cria todas as features técnicas."""
    if df.empty:
        raise ValueError("[ERRO] DataFrame está vazio")
    
    colunas_obrigatorias = ['abertura', 'maxima', 'minima', 'fechamento']
    if not all(col in df.columns for col in colunas_obrigatorias):
        raise KeyError(f"[ERRO] Colunas obrigatórias faltando")
    
    if verbose:
        print(f"[1/6] Criando features... Shape inicial: {df.shape}")
    
    df_features = df.copy()
    
    if incluir_retornos:
        df_features['returns'] = calcular_retornos_logaritmicos(df_features)
    
    if incluir_ema:
        for periodo in PERIODOS_EMA:
            df_features[f'ema_{periodo}'] = calcular_ema(df_features, periodo)
    
    if incluir_rsi:
        for periodo in PERIODOS_RSI:
            df_features[f'rsi_{periodo}'] = calcular_rsi(df_features, periodo)
    
    if incluir_bollinger:
        bollinger = calcular_bandas_bollinger(df_features)
        df_features = pd.concat([df_features, bollinger], axis=1)
    
    if incluir_volatilidade:
        df_features['volatility'] = calcular_volatilidade(df_features)
    
    if incluir_target and 'returns' in df_features.columns:
        df_features['target'] = criar_target_com_banda_morta(df_features)
        if verbose:
            n_alta = (df_features['target'] == 1).sum()
            n_baixa = (df_features['target'] == -1).sum()
            n_neutro = (df_features['target'] == 0).sum()
            total = len(df_features['target'].dropna())
            print(f"[OK] Target: Alta={n_alta} ({n_alta/total*100:.1f}%), "
                  f"Baixa={n_baixa} ({n_baixa/total*100:.1f}%), "
                  f"Neutro={n_neutro} ({n_neutro/total*100:.1f}%)")
    
    shape_antes = df_features.shape
    df_features = df_features.dropna()
    
    if verbose:
        print(f"[OK] Features criadas! Shape final: {df_features.shape} "
              f"(removidas {shape_antes[0] - df_features.shape[0]} linhas com NaN)")
    
    return df_features
