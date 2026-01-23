"""
Módulo para engenharia de features (indicadores técnicos).

Conforme metodologia do TCC (Seção 4.2 - Engenharia de Atributos).
Cria indicadores técnicos essenciais: retornos, MME, RSI, Bollinger, volatilidade
e gera target com banda morta para classificação.
"""

import pandas as pd
import numpy as np
from typing import Optional, List

try:
    from ..config import (
        PERIODOS_EMA,
        PERIODOS_RSI,
        PERIODO_BOLLINGER,
        DESVIOS_BOLLINGER,
        PERIODO_VOLATILIDADE,
        THRESHOLD_BANDA_MORTA
    )
except ImportError:
    # Para execução direta do script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import (
        PERIODOS_EMA,
        PERIODOS_RSI,
        PERIODO_BOLLINGER,
        DESVIOS_BOLLINGER,
        PERIODO_VOLATILIDADE,
        THRESHOLD_BANDA_MORTA
    )


def calcular_retornos_logaritmicos(df: pd.DataFrame, coluna_preco: str = 'fechamento') -> pd.Series:
    """
    Calcula retornos logarítmicos: rt = ln(Pt) - ln(Pt-1).
    
    Conforme metodologia do TCC (Seção 4.2).
    
    Parâmetros:
        df: DataFrame com dados OHLCV
        coluna_preco: Nome da coluna de preço (padrão: 'fechamento')
        
    Retorna:
        Series com retornos logarítmicos
        
    Exceções:
        KeyError: Se coluna_preco não existir no DataFrame
    """
    if coluna_preco not in df.columns:
        raise KeyError(f"[ERRO] Coluna '{coluna_preco}' não encontrada no DataFrame")
    
    retornos = np.log(df[coluna_preco] / df[coluna_preco].shift(1))
    return retornos


def calcular_ema(df: pd.DataFrame, periodo: int, coluna_preco: str = 'fechamento') -> pd.Series:
    """
    Calcula Média Móvel Exponencial (EMA).
    
    Parâmetros:
        df: DataFrame com dados OHLCV
        periodo: Período da EMA
        coluna_preco: Nome da coluna de preço
        
    Retorna:
        Series com valores da EMA
    """
    if coluna_preco not in df.columns:
        raise KeyError(f"[ERRO] Coluna '{coluna_preco}' não encontrada no DataFrame")
    
    ema = df[coluna_preco].ewm(span=periodo, adjust=False).mean()
    return ema


def calcular_rsi(df: pd.DataFrame, periodo: int = 14, coluna_preco: str = 'fechamento') -> pd.Series:
    """
    Calcula Relative Strength Index (RSI).
    
    Conforme metodologia do TCC (Seção 4.2).
    
    Parâmetros:
        df: DataFrame com dados OHLCV
        periodo: Período do RSI (padrão: 14)
        coluna_preco: Nome da coluna de preço
        
    Retorna:
        Series com valores do RSI (0-100)
    """
    if coluna_preco not in df.columns:
        raise KeyError(f"[ERRO] Coluna '{coluna_preco}' não encontrada no DataFrame")
    
    # Calcular variações
    delta = df[coluna_preco].diff()
    
    # Separar ganhos e perdas
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calcular médias móveis exponenciais de ganhos e perdas
    avg_gain = gain.ewm(span=periodo, adjust=False).mean()
    avg_loss = loss.ewm(span=periodo, adjust=False).mean()
    
    # Calcular RS e RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calcular_bandas_bollinger(
    df: pd.DataFrame,
    periodo: int = PERIODO_BOLLINGER,
    num_desvios: float = DESVIOS_BOLLINGER,
    coluna_preco: str = 'fechamento'
) -> pd.DataFrame:
    """
    Calcula Bandas de Bollinger.
    
    Conforme metodologia do TCC (Seção 4.2).
    
    Parâmetros:
        df: DataFrame com dados OHLCV
        periodo: Período para cálculo da média móvel (padrão: 20)
        num_desvios: Número de desvios-padrão (padrão: 2)
        coluna_preco: Nome da coluna de preço
        
    Retorna:
        DataFrame com colunas: 'bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position'
    """
    if coluna_preco not in df.columns:
        raise KeyError(f"[ERRO] Coluna '{coluna_preco}' não encontrada no DataFrame")
    
    # Calcular média móvel simples
    sma = df[coluna_preco].rolling(window=periodo).mean()
    
    # Calcular desvio-padrão
    std = df[coluna_preco].rolling(window=periodo).std()
    
    # Calcular bandas
    bb_upper = sma + (std * num_desvios)
    bb_lower = sma - (std * num_desvios)
    
    # Largura das bandas (normalizada)
    bb_width = (bb_upper - bb_lower) / sma
    
    # Posição relativa do preço dentro das bandas (0 = banda inferior, 1 = banda superior)
    bb_position = (df[coluna_preco] - bb_lower) / (bb_upper - bb_lower)
    
    resultado = pd.DataFrame({
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'bb_middle': sma,
        'bb_width': bb_width,
        'bb_position': bb_position
    })
    
    return resultado


def calcular_volatilidade(
    df: pd.DataFrame,
    periodo: int = PERIODO_VOLATILIDADE,
    coluna_retornos: str = 'returns'
) -> pd.Series:
    """
    Calcula volatilidade realizada (desvio-padrão dos retornos).
    
    Conforme metodologia do TCC (Seção 4.2).
    
    Parâmetros:
        df: DataFrame com coluna de retornos
        periodo: Janela móvel para cálculo (padrão: 20)
        coluna_retornos: Nome da coluna de retornos
        
    Retorna:
        Series com valores de volatilidade
    """
    if coluna_retornos not in df.columns:
        raise KeyError(f"[ERRO] Coluna '{coluna_retornos}' não encontrada no DataFrame")
    
    volatilidade = df[coluna_retornos].rolling(window=periodo).std()
    return volatilidade


def criar_target_com_banda_morta(
    df: pd.DataFrame,
    coluna_retornos: str = 'returns',
    threshold: float = THRESHOLD_BANDA_MORTA
) -> pd.Series:
    """
    Cria target de classificação com banda morta (dead band).
    
    Conforme metodologia do TCC (Seção 4.2 e 4.5).
    Movimentos menores que o threshold são considerados neutros (0).
    
    Parâmetros:
        df: DataFrame com coluna de retornos
        coluna_retornos: Nome da coluna de retornos
        threshold: Threshold para banda morta (padrão: 0.0005 = 0.05%)
        
    Retorna:
        Series com target: 1 (alta), -1 (baixa), 0 (neutro)
    """
    if coluna_retornos not in df.columns:
        raise KeyError(f"[ERRO] Coluna '{coluna_retornos}' não encontrada no DataFrame")
    
    # Retorno do próximo período (shift(-1) porque queremos prever o futuro)
    next_return = df[coluna_retornos].shift(-1)
    
    # Criar target
    target = pd.Series(0, index=df.index, dtype=int)  # Neutro por padrão
    
    # Alta: retorno > threshold
    target.loc[next_return > threshold] = 1
    
    # Baixa: retorno < -threshold
    target.loc[next_return < -threshold] = -1
    
    return target


def criar_features(
    df: pd.DataFrame,
    incluir_retornos: bool = True,
    incluir_ema: bool = True,
    incluir_rsi: bool = True,
    incluir_bollinger: bool = True,
    incluir_volatilidade: bool = True,
    incluir_target: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Cria todas as features técnicas para o modelo.
    
    Conforme metodologia do TCC (Seção 4.2 - Engenharia de Atributos).
    Orquestra o cálculo de todos os indicadores técnicos.
    
    Parâmetros:
        df: DataFrame com dados OHLCV (deve ter colunas: abertura, maxima, minima, fechamento)
        incluir_retornos: Se True, calcula retornos logarítmicos
        incluir_ema: Se True, calcula EMAs para períodos configurados
        incluir_rsi: Se True, calcula RSI para períodos configurados
        incluir_bollinger: Se True, calcula Bandas de Bollinger
        incluir_volatilidade: Se True, calcula volatilidade realizada
        incluir_target: Se True, cria target com banda morta
        verbose: Se True, exibe logs de progresso
        
    Retorna:
        DataFrame com todas as features adicionadas
        
    Exceções:
        KeyError: Se colunas obrigatórias estiverem faltando
        ValueError: Se DataFrame estiver vazio
    """
    if df.empty:
        raise ValueError("[ERRO] DataFrame está vazio")
    
    # Validar colunas obrigatórias
    colunas_obrigatorias = ['abertura', 'maxima', 'minima', 'fechamento']
    colunas_faltando = [col for col in colunas_obrigatorias if col not in df.columns]
    if colunas_faltando:
        raise KeyError(f"[ERRO] Colunas obrigatórias faltando: {colunas_faltando}")
    
    if verbose:
        print("[1/6] Iniciando criação de features...")
        print(f"      Shape inicial: {df.shape}")
    
    # Criar cópia para não modificar original
    df_features = df.copy()
    
    # [1/6] Retornos Logarítmicos
    if incluir_retornos:
        if verbose:
            print("[2/6] Calculando retornos logarítmicos...")
        df_features['returns'] = calcular_retornos_logaritmicos(df_features)
        if verbose:
            print(f"[OK] Retornos calculados. Missing: {df_features['returns'].isnull().sum()}")
    else:
        # Se não incluir retornos, criar coluna vazia para compatibilidade
        df_features['returns'] = np.nan
    
    # [2/6] Médias Móveis Exponenciais (EMA)
    if incluir_ema:
        if verbose:
            print(f"[3/6] Calculando EMAs para períodos {PERIODOS_EMA}...")
        for periodo in PERIODOS_EMA:
            coluna_nome = f'ema_{periodo}'
            df_features[coluna_nome] = calcular_ema(df_features, periodo)
        if verbose:
            print(f"[OK] {len(PERIODOS_EMA)} EMAs calculadas")
    
    # [3/6] Relative Strength Index (RSI)
    if incluir_rsi:
        if verbose:
            print(f"[4/6] Calculando RSI para períodos {PERIODOS_RSI}...")
        for periodo in PERIODOS_RSI:
            coluna_nome = f'rsi_{periodo}'
            df_features[coluna_nome] = calcular_rsi(df_features, periodo)
        if verbose:
            print(f"[OK] {len(PERIODOS_RSI)} RSIs calculados")
    
    # [4/6] Bandas de Bollinger
    if incluir_bollinger:
        if verbose:
            print(f"[5/6] Calculando Bandas de Bollinger (período={PERIODO_BOLLINGER}, desvios={DESVIOS_BOLLINGER})...")
        bollinger = calcular_bandas_bollinger(df_features)
        df_features = pd.concat([df_features, bollinger], axis=1)
        if verbose:
            print(f"[OK] Bandas de Bollinger calculadas. Colunas: {list(bollinger.columns)}")
    
    # [5/6] Volatilidade
    if incluir_volatilidade:
        if verbose:
            print(f"[6/6] Calculando volatilidade (janela={PERIODO_VOLATILIDADE})...")
        df_features['volatility'] = calcular_volatilidade(df_features)
        if verbose:
            print(f"[OK] Volatilidade calculada")
    
    # [6/6] Target com Banda Morta
    if incluir_target:
        if 'returns' not in df_features.columns or df_features['returns'].isnull().all():
            if verbose:
                print("[!] Retornos não disponíveis. Pulando criação de target.")
        else:
            if verbose:
                print(f"[7/6] Criando target com banda morta (threshold={THRESHOLD_BANDA_MORTA})...")
            df_features['target'] = criar_target_com_banda_morta(df_features)
            
            # Estatísticas do target
            n_alta = (df_features['target'] == 1).sum()
            n_baixa = (df_features['target'] == -1).sum()
            n_neutro = (df_features['target'] == 0).sum()
            total = len(df_features['target'].dropna())
            
            if verbose:
                print(f"[OK] Target criado:")
                print(f"      Alta: {n_alta} ({n_alta/total*100:.1f}%)")
                print(f"      Baixa: {n_baixa} ({n_baixa/total*100:.1f}%)")
                print(f"      Neutro: {n_neutro} ({n_neutro/total*100:.1f}%)")
    
    # Remover linhas com NaN (causadas por cálculos de janelas móveis)
    shape_antes = df_features.shape
    df_features = df_features.dropna()
    shape_depois = df_features.shape
    
    if verbose:
        n_removidas = shape_antes[0] - shape_depois[0]
        print(f"\n[OK] Features criadas! Shape final: {df_features.shape}")
        print(f"      Linhas removidas (NaN): {n_removidas}")
        print(f"      Colunas adicionadas: {df_features.shape[1] - df.shape[1]}")
    
    return df_features


def obter_lista_features() -> List[str]:
    """
    Retorna lista de todas as features que podem ser criadas.
    
    Útil para documentação e validação.
    
    Retorna:
        Lista com nomes das features
    """
    features = ['returns']
    
    # EMAs
    for periodo in PERIODOS_EMA:
        features.append(f'ema_{periodo}')
    
    # RSIs
    for periodo in PERIODOS_RSI:
        features.append(f'rsi_{periodo}')
    
    # Bollinger
    features.extend(['bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position'])
    
    # Volatilidade
    features.append('volatility')
    
    # Target
    features.append('target')
    
    return features


if __name__ == '__main__':
    """
    Teste básico do módulo.
    """
    import sys
    import os
    
    # Ajustar caminho para execução direta
    if __file__:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, script_dir)
    
    from data_processing.load_data import carregar_dados
    
    # Testar com um arquivo de exemplo
    arquivo_teste = 'data/raw/VALE3_M15_20200101_20251231.csv'
    
    if os.path.exists(arquivo_teste):
        print("=" * 70)
        print("TESTE DE FEATURE ENGINEERING")
        print("=" * 70)
        
        # Carregar dados
        print("\n[1/2] Carregando dados...")
        df = carregar_dados(arquivo_teste, verbose=False)
        print(f"[OK] Dados carregados. Shape: {df.shape}")
        
        # Criar features
        print("\n[2/2] Criando features...")
        df_features = criar_features(df, verbose=True)
        
        print("\n" + "=" * 70)
        print("RESUMO DAS FEATURES")
        print("=" * 70)
        print(f"\nColunas originais: {len(df.columns)}")
        print(f"Colunas após features: {len(df_features.columns)}")
        print(f"Features adicionadas: {len(df_features.columns) - len(df.columns)}")
        
        print("\nColunas criadas:")
        colunas_originais = set(df.columns)
        colunas_novas = [col for col in df_features.columns if col not in colunas_originais]
        for col in colunas_novas:
            missing = df_features[col].isnull().sum()
            print(f"  - {col}: {missing} missing values")
        
        print("\n" + "=" * 70)
        print("PRIMEIRAS 5 LINHAS (com features)")
        print("=" * 70)
        print(df_features.head())
        
        print("\n" + "=" * 70)
        print("ESTATÍSTICAS DESCRITIVAS")
        print("=" * 70)
        print(df_features[colunas_novas].describe())
        
    else:
        print(f"[ERRO] Arquivo não encontrado: {arquivo_teste}")
        print("\nUso: python feature_engineering.py")
        print("Ou: python feature_engineering.py <caminho_arquivo>")
