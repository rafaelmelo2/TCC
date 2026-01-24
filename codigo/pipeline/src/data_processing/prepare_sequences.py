"""Preparação de sequências temporais para modelos de deep learning."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..config import JANELA_TEMPORAL_STEPS


def selecionar_features_dl(df_features: pd.DataFrame) -> List[str]:
    """
    Seleciona features relevantes para deep learning.
    
    Features selecionadas (total: 12):
    - EMA: ema_9, ema_21, ema_50 (3)
    - RSI: rsi_9, rsi_21, rsi_50 (3)
    - Bollinger: bb_upper, bb_lower, bb_middle, bb_width, bb_position (5)
    - Volatilidade: volatility (1)
    
    Conforme metodologia do trabalho (Seção 4.3 - Arquitetura dos Modelos).
    
    Parâmetros:
        df_features: DataFrame com features criadas
    
    Retorna:
        Lista de nomes das colunas de features
    """
    features_dl = []
    
    # EMAs
    for periodo in [9, 21, 50]:
        col = f'ema_{periodo}'
        if col in df_features.columns:
            features_dl.append(col)
    
    # RSIs
    for periodo in [9, 21, 50]:
        col = f'rsi_{periodo}'
        if col in df_features.columns:
            features_dl.append(col)
    
    # Bollinger Bands
    bollinger_cols = ['bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position']
    for col in bollinger_cols:
        if col in df_features.columns:
            features_dl.append(col)
    
    # Volatilidade
    if 'volatility' in df_features.columns:
        features_dl.append('volatility')
    
    return features_dl


def criar_sequencias_temporais(X: np.ndarray, y: np.ndarray, 
                               n_steps: int = JANELA_TEMPORAL_STEPS,
                               verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria sequências temporais (janelas deslizantes) para deep learning.
    
    Para cada índice i, cria uma janela de n_steps barras anteriores:
    - X[i] = [X[i-n_steps], X[i-n_steps+1], ..., X[i-1]]
    - y[i] = target na posição i
    
    Conforme metodologia do trabalho (Seção 4.3 - Arquitetura dos Modelos).
    
    Parâmetros:
        X: Array 2D com features (n_samples, n_features)
        y: Array 1D com targets (n_samples,)
        n_steps: Número de barras históricas (janela temporal)
        verbose: Se True, imprime informações
    
    Retorna:
        X_seq: Array 3D (n_samples - n_steps, n_steps, n_features)
        y_seq: Array 1D (n_samples - n_steps,)
    
    Exceções:
        ValueError: Se X e y têm tamanhos incompatíveis
    """
    if len(X) != len(y):
        raise ValueError("[ERRO] X e y devem ter mesmo tamanho")
    
    if len(X) < n_steps + 1:
        raise ValueError(f"[ERRO] Dados insuficientes. Necessário pelo menos {n_steps + 1} amostras")
    
    n_samples = len(X) - n_steps
    n_features = X.shape[1]
    
    X_seq = np.zeros((n_samples, n_steps, n_features))
    y_seq = np.zeros(n_samples)
    
    for i in range(n_samples):
        X_seq[i] = X[i:i+n_steps]
        y_seq[i] = y[i+n_steps]
    
    if verbose:
        print(f"[OK] Sequências criadas: {X_seq.shape} (samples, timesteps, features)")
        print(f"     Target shape: {y_seq.shape}")
        print(f"     Janela temporal: {n_steps} barras")
    
    return X_seq, y_seq


def normalizar_features(X_train: np.ndarray, X_test: Optional[np.ndarray] = None,
                        metodo: str = 'minmax', verbose: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], object]:
    """
    Normaliza features usando dados de treino.
    
    CRÍTICO: Normalização ajustada APENAS nos dados de treino para evitar data leakage.
    Transformações do treino são aplicadas ao teste.
    
    Conforme metodologia do trabalho (Seção 4.2 - Prevenção de Data Leakage).
    
    Parâmetros:
        X_train: Features de treino (n_samples, n_features) ou (n_samples, n_steps, n_features)
        X_test: Features de teste (opcional, mesmo formato que X_train)
        metodo: 'minmax' ou 'standard' (z-score)
        verbose: Se True, imprime informações
    
    Retorna:
        X_train_norm: Features de treino normalizadas
        X_test_norm: Features de teste normalizadas (None se X_test não fornecido)
        scaler: Objeto scaler ajustado (para uso futuro)
    
    Exceções:
        ValueError: Se método de normalização inválido
    """
    if metodo not in ['minmax', 'standard']:
        raise ValueError("[ERRO] Método deve ser 'minmax' ou 'standard'")
    
    # Determinar se é 2D ou 3D
    is_3d = len(X_train.shape) == 3
    
    if is_3d:
        # Para sequências 3D: (n_samples, n_steps, n_features)
        # Reshape para 2D: (n_samples * n_steps, n_features)
        n_samples, n_steps, n_features = X_train.shape
        X_train_2d = X_train.reshape(-1, n_features)
        
        if X_test is not None:
            n_test_samples = X_test.shape[0]
            X_test_2d = X_test.reshape(-1, n_features)
        else:
            X_test_2d = None
    else:
        # Já é 2D: (n_samples, n_features)
        X_train_2d = X_train
        X_test_2d = X_test
    
    # Criar e ajustar scaler
    if metodo == 'minmax':
        scaler = MinMaxScaler()
    else:  # standard
        scaler = StandardScaler()
    
    X_train_norm_2d = scaler.fit_transform(X_train_2d)
    
    if X_test_2d is not None:
        X_test_norm_2d = scaler.transform(X_test_2d)
    else:
        X_test_norm_2d = None
    
    # Reshape de volta se era 3D
    if is_3d:
        X_train_norm = X_train_norm_2d.reshape(n_samples, n_steps, n_features)
        if X_test_norm_2d is not None:
            X_test_norm = X_test_norm_2d.reshape(n_test_samples, n_steps, n_features)
        else:
            X_test_norm = None
    else:
        X_train_norm = X_train_norm_2d
        X_test_norm = X_test_norm_2d
    
    if verbose:
        print(f"[OK] Features normalizadas ({metodo})")
        print(f"     Treino: {X_train_norm.shape}")
        if X_test_norm is not None:
            print(f"     Teste: {X_test_norm.shape}")
    
    return X_train_norm, X_test_norm, scaler


def preparar_dados_dl(df_features: pd.DataFrame,
                       train_start: int, train_end: int,
                       test_start: int, test_end: int,
                       n_steps: int = JANELA_TEMPORAL_STEPS,
                       metodo_normalizacao: str = 'minmax',
                       verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, List[str]]:
    """
    Prepara dados completos para deep learning: seleciona features, cria sequências e normaliza.
    
    Pipeline completo:
    1. Seleciona features relevantes
    2. Divide em treino/teste
    3. Cria sequências temporais
    4. Normaliza (ajustando apenas no treino)
    
    Conforme metodologia do trabalho (Seções 4.2 e 4.3).
    
    Parâmetros:
        df_features: DataFrame com features criadas (deve ter coluna 'target')
        train_start: Índice inicial do conjunto de treino
        train_end: Índice final do conjunto de treino (exclusivo)
        test_start: Índice inicial do conjunto de teste
        test_end: Índice final do conjunto de teste (exclusivo)
        n_steps: Número de barras históricas (janela temporal)
        metodo_normalizacao: 'minmax' ou 'standard'
        verbose: Se True, imprime informações
    
    Retorna:
        X_train_seq: Sequências de treino (n_samples, n_steps, n_features)
        y_train: Targets de treino (n_samples,)
        X_test_seq: Sequências de teste (n_samples, n_steps, n_features)
        y_test: Targets de teste (n_samples,)
        scaler: Objeto scaler ajustado
        feature_names: Lista de nomes das features usadas
    
    Exceções:
        ValueError: Se 'target' não está presente ou índices inválidos
    """
    if 'target' not in df_features.columns:
        raise ValueError("[ERRO] DataFrame deve ter coluna 'target'")
    
    if verbose:
        print(f"[1/4] Selecionando features para deep learning...")
    
    # Selecionar features
    feature_names = selecionar_features_dl(df_features)
    
    if len(feature_names) == 0:
        raise ValueError("[ERRO] Nenhuma feature encontrada para deep learning")
    
    if verbose:
        print(f"[OK] {len(feature_names)} features selecionadas: {feature_names[:5]}...")
    
    # Extrair features e target
    X_all = df_features[feature_names].values
    y_all = df_features['target'].values
    
    # Dividir em treino e teste
    X_train = X_all[train_start:train_end]
    y_train = y_all[train_start:train_end]
    X_test = X_all[test_start:test_end]
    y_test = y_all[test_start:test_end]
    
    if verbose:
        print(f"[2/4] Dados divididos:")
        print(f"     Treino: {X_train.shape} | Teste: {X_test.shape}")
    
    # Criar sequências temporais
    if verbose:
        print(f"[3/4] Criando sequências temporais...")
    
    X_train_seq, y_train_seq = criar_sequencias_temporais(
        X_train, y_train, n_steps=n_steps, verbose=verbose
    )
    
    X_test_seq, y_test_seq = criar_sequencias_temporais(
        X_test, y_test, n_steps=n_steps, verbose=verbose
    )
    
    # Normalizar (ajustando apenas no treino)
    if verbose:
        print(f"[4/4] Normalizando features ({metodo_normalizacao})...")
    
    X_train_norm, X_test_norm, scaler = normalizar_features(
        X_train_seq, X_test_seq, metodo=metodo_normalizacao, verbose=verbose
    )
    
    if verbose:
        print(f"[OK] Dados preparados para deep learning!")
        print(f"     Treino: {X_train_norm.shape}, {y_train_seq.shape}")
        print(f"     Teste: {X_test_norm.shape}, {y_test_seq.shape}")
    
    return X_train_norm, y_train_seq, X_test_norm, y_test_seq, scaler, feature_names
