"""Otimização bayesiana de hiperparâmetros usando Optuna."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, Callable, Optional, Tuple
import optuna
from optuna.trial import Trial

from ..config import SEED, HIPERPARAMETROS_LSTM, HIPERPARAMETROS_CNN_LSTM
from ..models.lstm_model import criar_modelo_lstm
from ..models.cnn_lstm_model import criar_modelo_cnn_lstm

# Fixar seed para reprodutibilidade
np.random.seed(SEED)
tf.random.set_seed(SEED)


def criar_espaco_busca_lstm(trial: Trial) -> Dict[str, Any]:
    """
    Cria espaço de busca de hiperparâmetros para LSTM.
    
    Parâmetros:
        trial: Trial do Optuna
    
    Retorna:
        Dicionário com hiperparâmetros sugeridos
    """
    return {
        'lstm_units': trial.suggest_categorical('lstm_units', HIPERPARAMETROS_LSTM['lstm_units']),
        'dropout': trial.suggest_categorical('dropout', HIPERPARAMETROS_LSTM['dropout']),
        'learning_rate': trial.suggest_categorical('learning_rate', HIPERPARAMETROS_LSTM['learning_rate']),
        'batch_size': trial.suggest_categorical('batch_size', HIPERPARAMETROS_LSTM['batch_size'])
    }


def criar_espaco_busca_cnn_lstm(trial: Trial) -> Dict[str, Any]:
    """
    Cria espaço de busca de hiperparâmetros para CNN-LSTM.
    
    Parâmetros:
        trial: Trial do Optuna
    
    Retorna:
        Dicionário com hiperparâmetros sugeridos
    """
    return {
        'conv_filters': trial.suggest_categorical('conv_filters', HIPERPARAMETROS_CNN_LSTM['conv_filters']),
        'conv_kernel_size': trial.suggest_categorical('conv_kernel_size', HIPERPARAMETROS_CNN_LSTM['conv_kernel_size']),
        'pool_size': HIPERPARAMETROS_CNN_LSTM['pool_size'][0],  # Fixo
        'lstm_units': trial.suggest_categorical('lstm_units', HIPERPARAMETROS_CNN_LSTM['lstm_units']),
        'dropout': trial.suggest_categorical('dropout', HIPERPARAMETROS_CNN_LSTM['dropout']),
        'learning_rate': trial.suggest_categorical('learning_rate', HIPERPARAMETROS_CNN_LSTM['learning_rate']),
        'batch_size': trial.suggest_categorical('batch_size', HIPERPARAMETROS_CNN_LSTM['batch_size'])
    }


def objetivo_lstm(trial: Trial, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  n_steps: int, n_features: int,
                  epochs: int = 30, verbose: int = 0) -> float:
    """
    Função objetivo para otimização de LSTM.
    
    IMPORTANTE: A otimização é feita no conjunto de VALIDAÇÃO interno,
    não no conjunto de teste. Isso previne data leakage.
    
    Parâmetros:
        trial: Trial do Optuna
        X_train: Sequências de treino
        y_train: Targets de treino
        X_val: Sequências de validação (usado para otimização)
        y_val: Targets de validação (usado para otimização)
        n_steps: Número de barras históricas
        n_features: Número de features
        epochs: Número máximo de épocas
        verbose: Verbosidade
    
    Retorna:
        Acurácia direcional no conjunto de validação (a ser maximizada)
    """
    # Resetar seeds para cada trial (usando número do trial para variar)
    trial_seed = SEED + trial.number
    np.random.seed(trial_seed)
    tf.random.set_seed(trial_seed)
    
    # Obter hiperparâmetros sugeridos
    hiperparams = criar_espaco_busca_lstm(trial)
    
    # Criar modelo com hiperparâmetros sugeridos
    model = criar_modelo_lstm(
        n_steps=n_steps,
        n_features=n_features,
        lstm_units=hiperparams['lstm_units'],
        dropout=hiperparams['dropout'],
        learning_rate=hiperparams['learning_rate']
    )
    
    # Converter targets para binário e remover neutros
    # IMPORTANTE: Remover neutros para evitar desbalanceamento
    mask_train = y_train != 0
    mask_val = y_val != 0
    
    y_train_filtered = y_train[mask_train]
    y_val_filtered = y_val[mask_val]
    X_train_filtered = X_train[mask_train]
    X_val_filtered = X_val[mask_val]
    
    y_train_binary = np.where(y_train_filtered == 1, 1, 0)
    y_val_binary = np.where(y_val_filtered == 1, 1, 0)
    
    # Calcular class weights para balancear (se houver desbalanceamento)
    n_class_0 = np.sum(y_train_binary == 0)  # Baixa
    n_class_1 = np.sum(y_train_binary == 1)  # Alta
    total = len(y_train_binary)
    
    if n_class_0 > 0 and n_class_1 > 0:
        # Calcular pesos inversamente proporcionais à frequência
        weight_0 = total / (2 * n_class_0)
        weight_1 = total / (2 * n_class_1)
        class_weight = {0: weight_0, 1: weight_1}
    else:
        class_weight = None
    
    # Callbacks para early stopping
    # Reduzir patience para permitir mais variação entre trials
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,  # Reduzido de 10 para 5 para permitir mais variação
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,  # Reduzido de 5 para 3
            min_lr=1e-7,
            verbose=0
        )
    ]
    
    # Treinar modelo
    try:
        # Limpar sessão do TensorFlow entre trials para evitar vazamento
        keras.backend.clear_session()
        
        history = model.fit(
            X_train_filtered, y_train_binary,
            validation_data=(X_val_filtered, y_val_binary),
            epochs=epochs,
            batch_size=hiperparams['batch_size'],
            callbacks=callbacks_list,
            class_weight=class_weight,  # Usar class weights para balancear
            verbose=verbose
        )
        
        # Fazer previsões no conjunto de validação
        y_pred_proba = model.predict(X_val_filtered, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Converter de binário para direção (-1, 1)
        y_pred_direcao = np.where(y_pred == 1, 1, -1)
        
        # Calcular acurácia direcional (comparar com y_val_filtered que não tem neutros)
        acuracia = np.mean(y_pred_direcao == y_val_filtered)
        
        return acuracia
    
    except Exception as e:
        # Se houver erro, retornar valor muito baixo
        if verbose > 0:
            print(f"     Erro no trial {trial.number}: {e}")
        return 0.0


def objetivo_cnn_lstm(trial: Trial, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     n_steps: int, n_features: int,
                     epochs: int = 30, verbose: int = 0) -> float:
    """
    Função objetivo para otimização de CNN-LSTM.
    
    IMPORTANTE: A otimização é feita no conjunto de VALIDAÇÃO interno,
    não no conjunto de teste. Isso previne data leakage.
    
    Parâmetros:
        trial: Trial do Optuna
        X_train: Sequências de treino
        y_train: Targets de treino
        X_val: Sequências de validação (usado para otimização)
        y_val: Targets de validação (usado para otimização)
        n_steps: Número de barras históricas
        n_features: Número de features
        epochs: Número máximo de épocas
        verbose: Verbosidade
    
    Retorna:
        Acurácia direcional no conjunto de validação (a ser maximizada)
    """
    # Resetar seeds para cada trial (usando número do trial para variar)
    trial_seed = SEED + trial.number
    np.random.seed(trial_seed)
    tf.random.set_seed(trial_seed)
    
    # Obter hiperparâmetros sugeridos
    hiperparams = criar_espaco_busca_cnn_lstm(trial)
    
    # Criar modelo com hiperparâmetros sugeridos
    model = criar_modelo_cnn_lstm(
        n_steps=n_steps,
        n_features=n_features,
        conv_filters=hiperparams['conv_filters'],
        conv_kernel_size=hiperparams['conv_kernel_size'],
        pool_size=hiperparams['pool_size'],
        lstm_units=hiperparams['lstm_units'],
        dropout=hiperparams['dropout'],
        learning_rate=hiperparams['learning_rate']
    )
    
    # Converter targets para binário e remover neutros
    # IMPORTANTE: Remover neutros para evitar desbalanceamento
    mask_train = y_train != 0
    mask_val = y_val != 0
    
    y_train_filtered = y_train[mask_train]
    y_val_filtered = y_val[mask_val]
    X_train_filtered = X_train[mask_train]
    X_val_filtered = X_val[mask_val]
    
    y_train_binary = np.where(y_train_filtered == 1, 1, 0)
    y_val_binary = np.where(y_val_filtered == 1, 1, 0)
    
    # Calcular class weights para balancear (se houver desbalanceamento)
    n_class_0 = np.sum(y_train_binary == 0)  # Baixa
    n_class_1 = np.sum(y_train_binary == 1)  # Alta
    total = len(y_train_binary)
    
    if n_class_0 > 0 and n_class_1 > 0:
        # Calcular pesos inversamente proporcionais à frequência
        weight_0 = total / (2 * n_class_0)
        weight_1 = total / (2 * n_class_1)
        class_weight = {0: weight_0, 1: weight_1}
    else:
        class_weight = None
    
    # Callbacks para early stopping
    # Reduzir patience para permitir mais variação entre trials
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,  # Reduzido de 10 para 5 para permitir mais variação
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,  # Reduzido de 5 para 3
            min_lr=1e-7,
            verbose=0
        )
    ]
    
    # Treinar modelo
    try:
        # Limpar sessão do TensorFlow entre trials para evitar vazamento
        keras.backend.clear_session()
        
        history = model.fit(
            X_train_filtered, y_train_binary,
            validation_data=(X_val_filtered, y_val_binary),
            epochs=epochs,
            batch_size=hiperparams['batch_size'],
            callbacks=callbacks_list,
            class_weight=class_weight,  # Usar class weights para balancear
            verbose=verbose
        )
        
        # Fazer previsões no conjunto de validação
        y_pred_proba = model.predict(X_val_filtered, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Converter de binário para direção (-1, 1)
        y_pred_direcao = np.where(y_pred == 1, 1, -1)
        
        # Calcular acurácia direcional (comparar com y_val_filtered que não tem neutros)
        acuracia = np.mean(y_pred_direcao == y_val_filtered)
        
        # Debug: verificar se o modelo está variando
        if verbose > 0 or trial.number % 5 == 0:
            n_pred_1 = np.sum(y_pred_direcao == 1)
            n_pred_neg1 = np.sum(y_pred_direcao == -1)
            n_val_1 = np.sum(y_val_filtered == 1)
            n_val_neg1 = np.sum(y_val_filtered == -1)
            pred_mean = np.mean(y_pred_proba)
            pred_min = np.min(y_pred_proba)
            pred_max = np.max(y_pred_proba)
            pred_std = np.std(y_pred_proba)
            print(f"     Trial {trial.number}: Pred=[1:{n_pred_1}, -1:{n_pred_neg1}], "
                  f"Val=[1:{n_val_1}, -1:{n_val_neg1}], "
                  f"Proba=[{pred_min:.3f}-{pred_max:.3f}, mean={pred_mean:.3f}, std={pred_std:.3f}], "
                  f"Acc={acuracia:.4f}")
        
        return acuracia
    
    except Exception as e:
        # Se houver erro, retornar valor muito baixo
        if verbose > 0:
            print(f"     Erro no trial {trial.number}: {e}")
        return 0.0


def otimizar_hiperparametros(X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             modelo_tipo: str,
                             n_steps: int, n_features: int,
                             n_trials: int = 20,
                             epochs: int = 30,
                             verbose: bool = True) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Otimiza hiperparâmetros usando Optuna.
    
    METODOLOGIA:
    - Otimização bayesiana (Tree-structured Parzen Estimator)
    - Otimização feita no conjunto de VALIDAÇÃO interno
    - Cada fold do walk-forward tem sua própria otimização
    - Previne data leakage: teste nunca é usado para otimização
    
    Conforme metodologia do TCC (Seção 4.4.2 - Seleção de Hiperparâmetros).
    
    Parâmetros:
        X_train: Sequências de treino
        y_train: Targets de treino
        X_val: Sequências de validação (usado para otimização)
        y_val: Targets de validação (usado para otimização)
        modelo_tipo: 'lstm' ou 'cnn_lstm'
        n_steps: Número de barras históricas
        n_features: Número de features
        n_trials: Número de trials do Optuna (padrão: 20)
        epochs: Número máximo de épocas por trial
        verbose: Se True, imprime progresso
    
    Retorna:
        Tupla (melhores_hiperparametros, study)
    """
    # Escolher função objetivo
    if modelo_tipo == 'lstm':
        objetivo = lambda trial: objetivo_lstm(
            trial, X_train, y_train, X_val, y_val,
            n_steps, n_features, epochs, verbose=0
        )
    elif modelo_tipo == 'cnn_lstm':
        objetivo = lambda trial: objetivo_cnn_lstm(
            trial, X_train, y_train, X_val, y_val,
            n_steps, n_features, epochs, verbose=0
        )
    else:
        raise ValueError(f"[ERRO] Tipo de modelo inválido: {modelo_tipo}")
    
    # Criar study do Optuna
    study = optuna.create_study(
        direction='maximize',  # Maximizar acurácia direcional
        sampler=optuna.samplers.TPESampler(seed=SEED),  # Tree-structured Parzen Estimator
        study_name=f'optuna_{modelo_tipo}'
    )
    
    if verbose:
        print(f"     Otimizando {modelo_tipo.upper()} com {n_trials} trials...")
    
    # Executar otimização
    study.optimize(objetivo, n_trials=n_trials, show_progress_bar=verbose)
    
    # Obter melhores hiperparâmetros
    melhores_hiperparams = study.best_params
    
    if verbose:
        print(f"     Melhor acurácia (validação): {study.best_value:.4f}")
        print(f"     Melhores hiperparâmetros: {melhores_hiperparams}")
    
    return melhores_hiperparams, study
