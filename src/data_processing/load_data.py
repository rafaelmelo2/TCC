"""
Módulo para carregamento e validação de dados intradiários da B3.

Conforme metodologia do TCC (Seção 4.1 - Aquisição de Dados).
Valida estrutura, timestamps, horário de pregão e remove dados inválidos.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple

try:
    from ..config import (
        COLUNAS_OBRIGATORIAS,
        HORARIO_ABERTURA,
        HORARIO_FECHAMENTO
    )
except ImportError:
    # Para execução direta do script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import (
        COLUNAS_OBRIGATORIAS,
        HORARIO_ABERTURA,
        HORARIO_FECHAMENTO
    )


def carregar_dados(
    caminho_arquivo: str,
    validar_pregão: bool = True,
    remover_volume_zero: bool = True,
    remover_duplicatas: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Carrega e valida dados intradiários de ações da B3.
    
    Conforme metodologia do TCC (Seção 4.1). Valida estrutura dos dados,
    timestamps, horário de pregão e remove barras inválidas.
    
    Parâmetros:
        caminho_arquivo: Caminho para o arquivo CSV com dados OHLCV
        validar_pregão: Se True, remove barras fora do horário de pregão (10h-17h)
        remover_volume_zero: Se True, remove barras com volume zero
        remover_duplicatas: Se True, remove timestamps duplicados (mantém primeiro)
        verbose: Se True, exibe logs de progresso
        
    Retorna:
        DataFrame com dados validados e limpos, indexado por timestamp
        
    Exceções:
        FileNotFoundError: Se o arquivo não existir
        ValueError: Se estrutura de dados estiver incorreta
        KeyError: Se colunas obrigatórias estiverem faltando
        
    Exemplo:
        >>> df = carregar_dados('data/raw/PETR4_M15_20200101_20251231.csv')
        >>> print(f"Shape: {df.shape}")
        >>> print(f"Período: {df.index.min()} até {df.index.max()}")
    """
    # [1/7] Validar existência do arquivo
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"[ERRO] Arquivo não encontrado: {caminho_arquivo}")
    
    if verbose:
        print(f"[1/7] Carregando arquivo: {os.path.basename(caminho_arquivo)}")
    
    # [2/7] Carregar CSV
    try:
        df = pd.read_csv(caminho_arquivo, low_memory=False)
        if verbose:
            print(f"[OK] Arquivo carregado. Shape inicial: {df.shape}")
    except Exception as e:
        raise ValueError(f"[ERRO] Erro ao ler CSV: {str(e)}")
    
    # [3/7] Validar colunas obrigatórias
    colunas_faltando = [col for col in COLUNAS_OBRIGATORIAS if col not in df.columns]
    if colunas_faltando:
        raise KeyError(
            f"[ERRO] Colunas obrigatórias faltando: {colunas_faltando}\n"
            f"Colunas disponíveis: {df.columns.tolist()}"
        )
    
    if verbose:
        print(f"[2/7] Colunas validadas: {len(COLUNAS_OBRIGATORIAS)} obrigatórias presentes")
    
    # [4/7] Converter timestamp e definir como índice
    try:
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        df = df.dropna(subset=['data'])  # Remove linhas com data inválida
        
        if len(df) == 0:
            raise ValueError("[ERRO] Nenhuma data válida encontrada após conversão")
        
        # Definir timestamp como índice
        df = df.set_index('data')
        df.index.name = 'timestamp'
        
        if verbose:
            print(f"[3/7] Timestamps convertidos. Período: {df.index.min()} até {df.index.max()}")
    except Exception as e:
        raise ValueError(f"[ERRO] Erro ao processar timestamps: {str(e)}")
    
    # [5/7] Validar ordem cronológica
    if not df.index.is_monotonic_increasing:
        if verbose:
            print("[!] Dados não estão em ordem cronológica. Ordenando...")
        df = df.sort_index()
    
    # Verificar duplicatas de timestamp
    duplicatas = df.index.duplicated()
    if duplicatas.any():
        n_duplicatas = duplicatas.sum()
        if verbose:
            print(f"[!] Encontradas {n_duplicatas} barras com timestamp duplicado")
        
        if remover_duplicatas:
            df = df[~duplicatas]
            if verbose:
                print(f"[OK] Duplicatas removidas. Shape atual: {df.shape}")
        else:
            if verbose:
                print("[!] Duplicatas mantidas (remover_duplicatas=False)")
    
    # [6/7] Validar horário de pregão (10:00 - 17:00)
    if validar_pregão:
        # Extrair apenas a hora do timestamp
        hora_series = pd.Series(df.index.time, index=df.index)
        mask_pregão = (
            (hora_series >= HORARIO_ABERTURA) & 
            (hora_series <= HORARIO_FECHAMENTO)
        )
        
        n_fora_pregão = (~mask_pregão).sum()
        if n_fora_pregão > 0:
            if verbose:
                print(f"[!] Encontradas {n_fora_pregão} barras fora do horário de pregão (10h-17h)")
            df = df[mask_pregão]
            if verbose:
                print(f"[OK] Barras fora do pregão removidas. Shape atual: {df.shape}")
        else:
            if verbose:
                print("[OK] Todas as barras estão dentro do horário de pregão")
    
    # [7/7] Validar e limpar dados OHLCV
    # Remover linhas com valores NaN em colunas críticas
    mask_nan = df[['abertura', 'maxima', 'minima', 'fechamento']].isnull().any(axis=1)
    n_nan = mask_nan.sum()
    if n_nan > 0:
        if verbose:
            print(f"[!] Encontradas {n_nan} barras com valores NaN em OHLC")
        df = df[~mask_nan]
        if verbose:
            print(f"[OK] Barras com NaN removidas. Shape atual: {df.shape}")
    
    # Validar lógica OHLC (high >= low, high >= open, high >= close, etc.)
    mask_invalido = (
        (df['maxima'] < df['minima']) |
        (df['maxima'] < df['abertura']) |
        (df['maxima'] < df['fechamento']) |
        (df['minima'] > df['abertura']) |
        (df['minima'] > df['fechamento']) |
        (df[['abertura', 'maxima', 'minima', 'fechamento']] <= 0).any(axis=1)
    )
    
    n_invalidos = mask_invalido.sum()
    if n_invalidos > 0:
        if verbose:
            print(f"[!] Encontradas {n_invalidos} barras com valores OHLC inválidos")
        df = df[~mask_invalido]
        if verbose:
            print(f"[OK] Barras inválidas removidas. Shape atual: {df.shape}")
    
    # Remover barras com volume zero
    if remover_volume_zero:
        mask_volume_zero = (df['volume_real'] == 0) | (df['volume_real'].isnull())
        n_volume_zero = mask_volume_zero.sum()
        if n_volume_zero > 0:
            if verbose:
                print(f"[!] Encontradas {n_volume_zero} barras com volume zero")
            df = df[~mask_volume_zero]
            if verbose:
                print(f"[OK] Barras com volume zero removidas. Shape atual: {df.shape}")
    
    # Renomear colunas para padrão (opcional, manter original se preferir)
    # df = df.rename(columns={
    #     'abertura': 'open',
    #     'maxima': 'high',
    #     'minima': 'low',
    #     'fechamento': 'close',
    #     'volume_real': 'volume'
    # })
    
    # Validar período final
    if len(df) == 0:
        raise ValueError("[ERRO] Nenhuma barra válida restou após validação")
    
    if verbose:
        print(f"[OK] Validação completa! Shape final: {df.shape}")
        print(f"[OK] Período final: {df.index.min()} até {df.index.max()}")
        print(f"[OK] Total de barras: {len(df):,}")
        print(f"[OK] Missing values: {df.isnull().sum().sum()}")
    
    return df


def validar_estrutura_dados(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Valida estrutura básica de um DataFrame de dados financeiros.
    
    Parâmetros:
        df: DataFrame a ser validado (deve ter timestamp como índice)
        
    Retorna:
        Tupla (é_válido, lista_de_erros)
    """
    erros = []
    
    # Verificar colunas obrigatórias (exceto 'data' que vira índice)
    colunas_obrigatorias_sem_data = [col for col in COLUNAS_OBRIGATORIAS if col != 'data']
    colunas_faltando = [col for col in colunas_obrigatorias_sem_data if col not in df.columns]
    if colunas_faltando:
        erros.append(f"Colunas faltando: {colunas_faltando}")
    
    # Verificar se índice é DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        erros.append("Índice não é DatetimeIndex")
    
    # Verificar ordem cronológica
    if not df.index.is_monotonic_increasing:
        erros.append("Dados não estão em ordem cronológica")
    
    # Verificar duplicatas
    if df.index.duplicated().any():
        erros.append(f"Encontradas {df.index.duplicated().sum()} timestamps duplicados")
    
    # Verificar missing values em colunas críticas
    for col in ['abertura', 'maxima', 'minima', 'fechamento']:
        if col in df.columns and df[col].isnull().any():
            n_missing = df[col].isnull().sum()
            erros.append(f"Coluna '{col}' tem {n_missing} valores NaN")
    
    return len(erros) == 0, erros


def obter_estatisticas_dados(df: pd.DataFrame) -> dict:
    """
    Retorna estatísticas descritivas dos dados carregados.
    
    Parâmetros:
        df: DataFrame com dados OHLCV
        
    Retorna:
        Dicionário com estatísticas (shape, período, missing, etc.)
    """
    stats = {
        'shape': df.shape,
        'período_início': df.index.min(),
        'período_fim': df.index.max(),
        'total_barras': len(df),
        'dias_úteis': (df.index.max() - df.index.min()).days,
        'missing_values': df.isnull().sum().to_dict(),
        'volume_total': df['volume_real'].sum() if 'volume_real' in df.columns else None,
        'preço_médio': df['fechamento'].mean() if 'fechamento' in df.columns else None,
        'preço_min': df['fechamento'].min() if 'fechamento' in df.columns else None,
        'preço_max': df['fechamento'].max() if 'fechamento' in df.columns else None,
    }
    
    return stats


if __name__ == '__main__':
    """
    Teste básico do módulo.
    """
    import sys
    
    # Testar com um arquivo de exemplo
    if len(sys.argv) > 1:
        arquivo_teste = sys.argv[1]
    else:
        # Tentar encontrar um arquivo de exemplo
        arquivo_teste = 'data/raw/VALE3_M15_20200101_20251231.csv'
    
    if os.path.exists(arquivo_teste):
        print("=" * 60)
        print("TESTE DE CARREGAMENTO DE DADOS")
        print("=" * 60)
        
        df = carregar_dados(arquivo_teste, verbose=True)
        
        print("\n" + "=" * 60)
        print("ESTATÍSTICAS DOS DADOS")
        print("=" * 60)
        stats = obter_estatisticas_dados(df)
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print("\n" + "=" * 60)
        print("VALIDAÇÃO DE ESTRUTURA")
        print("=" * 60)
        é_válido, erros = validar_estrutura_dados(df)
        if é_válido:
            print("[OK] Estrutura de dados válida!")
        else:
            print("[ERRO] Problemas encontrados:")
            for erro in erros:
                print(f"  - {erro}")
        
        print("\n" + "=" * 60)
        print("PRIMEIRAS 5 LINHAS")
        print("=" * 60)
        print(df.head())
        
        print("\n" + "=" * 60)
        print("ÚLTIMAS 5 LINHAS")
        print("=" * 60)
        print(df.tail())
    else:
        print(f"[ERRO] Arquivo não encontrado: {arquivo_teste}")
        print("Uso: python load_data.py <caminho_arquivo>")
