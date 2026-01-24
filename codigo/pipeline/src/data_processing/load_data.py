"""Carregamento e validação de dados intradiários da B3."""

import os
import pandas as pd
import numpy as np

from ..config import COLUNAS_OBRIGATORIAS, HORARIO_ABERTURA, HORARIO_FECHAMENTO


def carregar_dados(caminho_arquivo: str, validar_pregão: bool = True, remover_volume_zero: bool = True, 
                   remover_duplicatas: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Carrega e valida dados intradiários de ações da B3."""
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"[ERRO] Arquivo não encontrado: {caminho_arquivo}")
    
    if verbose:
        print(f"[1/5] Carregando: {os.path.basename(caminho_arquivo)}")
    
    df = pd.read_csv(caminho_arquivo, low_memory=False)
    if verbose:
        print(f"[OK] Shape inicial: {df.shape}")
    
    colunas_faltando = [col for col in COLUNAS_OBRIGATORIAS if col not in df.columns]
    if colunas_faltando:
        raise KeyError(f"[ERRO] Colunas faltando: {colunas_faltando}")
    
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    df = df.dropna(subset=['data'])
    if len(df) == 0:
        raise ValueError("[ERRO] Nenhuma data válida encontrada")
    
    df = df.set_index('data')
    df.index.name = 'timestamp'
    
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    
    if df.index.duplicated().any():
        n_dup = df.index.duplicated().sum()
        if verbose:
            print(f"[!] {n_dup} timestamps duplicados")
        if remover_duplicatas:
            df = df[~df.index.duplicated()]
    
    if validar_pregão:
        hora_series = pd.Series(df.index.time, index=df.index)
        mask = (hora_series >= HORARIO_ABERTURA) & (hora_series <= HORARIO_FECHAMENTO)
        n_fora = (~mask).sum()
        if n_fora > 0:
            if verbose:
                print(f"[!] {n_fora} barras fora do pregão")
            df = df[mask]
    
    mask_nan = df[['abertura', 'maxima', 'minima', 'fechamento']].isnull().any(axis=1)
    if mask_nan.any():
        df = df[~mask_nan]
    
    mask_invalido = (
        (df['maxima'] < df['minima']) |
        (df['maxima'] < df['abertura']) |
        (df['maxima'] < df['fechamento']) |
        (df['minima'] > df['abertura']) |
        (df['minima'] > df['fechamento']) |
        (df[['abertura', 'maxima', 'minima', 'fechamento']] <= 0).any(axis=1)
    )
    if mask_invalido.any():
        df = df[~mask_invalido]
    
    if remover_volume_zero:
        mask_volume = (df['volume_real'] == 0) | (df['volume_real'].isnull())
        if mask_volume.any():
            df = df[~mask_volume]
    
    if len(df) == 0:
        raise ValueError("[ERRO] Nenhuma barra válida restou")
    
    if verbose:
        print(f"[OK] Shape final: {df.shape} | Período: {df.index.min()} até {df.index.max()}")
    
    return df


def validar_estrutura_dados(df: pd.DataFrame) -> tuple:
    """Valida estrutura básica do DataFrame."""
    erros = []
    
    colunas_obrigatorias_sem_data = [col for col in COLUNAS_OBRIGATORIAS if col != 'data']
    colunas_faltando = [col for col in colunas_obrigatorias_sem_data if col not in df.columns]
    if colunas_faltando:
        erros.append(f"Colunas faltando: {colunas_faltando}")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        erros.append("Índice não é DatetimeIndex")
    
    if not df.index.is_monotonic_increasing:
        erros.append("Dados não estão em ordem cronológica")
    
    if df.index.duplicated().any():
        erros.append(f"{df.index.duplicated().sum()} timestamps duplicados")
    
    for col in ['abertura', 'maxima', 'minima', 'fechamento']:
        if col in df.columns and df[col].isnull().any():
            erros.append(f"'{col}' tem {df[col].isnull().sum()} NaN")
    
    return len(erros) == 0, erros


def obter_estatisticas_dados(df: pd.DataFrame) -> dict:
    """Retorna estatísticas descritivas dos dados."""
    return {
        'shape': df.shape,
        'período_início': df.index.min(),
        'período_fim': df.index.max(),
        'total_barras': len(df),
        'dias_úteis': (df.index.max() - df.index.min()).days,
        'missing_values': df.isnull().sum().to_dict(),
        'volume_total': df['volume_real'].sum() if 'volume_real' in df.columns else None,
        'preço_médio': df['fechamento'].mean() if 'fechamento' in df.columns else None,
    }
