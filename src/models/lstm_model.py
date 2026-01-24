"""Modelo LSTM puro (Baseline 3) para comparação."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..config import SEED

# Fixar seed para reprodutibilidade
tf.random.set_seed(SEED)


def criar_modelo_lstm(n_steps: int, n_features: int, 
                      lstm_units: int = 50, dropout: float = 0.2,
                      learning_rate: float = 0.001) -> keras.Model:
    """
    Cria modelo LSTM puro (Baseline 3).
    
    Este é um modelo baseline para comparação com o modelo híbrido CNN-LSTM.
    Usa apenas camadas LSTM para capturar dependências temporais de longo prazo.
    
    Arquitetura:
    - LSTM: Camada recorrente com unidades especificadas
    - Dropout: Regularização para evitar overfitting
    - Dense(1, sigmoid): Camada de saída para classificação binária
    
    Conforme metodologia do trabalho (Seção 4.3 - Arquitetura dos Modelos).
    
    Parâmetros:
        n_steps: Número de barras históricas (janela temporal)
        n_features: Número de features de entrada
        lstm_units: Número de unidades na camada LSTM (padrão: 50)
        dropout: Taxa de dropout para regularização (padrão: 0.2)
        learning_rate: Taxa de aprendizado do otimizador (padrão: 0.001)
    
    Retorna:
        Modelo Keras compilado e pronto para treinamento
    
    Exemplo:
        >>> model = criar_modelo_lstm(n_steps=60, n_features=12)
        >>> model.summary()
    """
    model = keras.Sequential([
        layers.LSTM(
            lstm_units, 
            dropout=dropout, 
            input_shape=(n_steps, n_features),
            name='lstm_layer'
        ),
        layers.Dense(1, activation='sigmoid', name='output_layer')
    ])
    
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
