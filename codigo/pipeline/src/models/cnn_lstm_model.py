"""Modelo híbrido CNN-LSTM (Modelo Principal) para predição de direção de preços."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..config import SEED

# Fixar seed para reprodutibilidade
tf.random.set_seed(SEED)


def criar_modelo_cnn_lstm(n_steps: int, n_features: int,
                         conv_filters: int = 64, conv_kernel_size: int = 2,
                         pool_size: int = 2, lstm_units: int = 50,
                         dropout: float = 0.2, learning_rate: float = 0.001) -> keras.Model:
    """
    Cria modelo híbrido CNN-LSTM (Modelo Principal do TCC).
    
    Este é o modelo principal proposto no trabalho, combinando:
    - CNN (Convolutional Neural Network): Para extrair padrões locais em janelas curtas
    - LSTM (Long Short-Term Memory): Para capturar dependências temporais de longo prazo
    
    Arquitetura completa:
    1. Conv1D: Extrai padrões locais através de convolução 1D
    2. MaxPooling1D: Reduz dimensionalidade e aumenta robustez
    3. LSTM: Captura dependências temporais de longo prazo
    4. Dense(1, sigmoid): Classificação binária (direção: alta/baixa)
    
    Justificativa da arquitetura:
    - CNN captura padrões de curto prazo (ex: reversão em 2-3 barras)
    - LSTM captura tendências e padrões de longo prazo (ex: tendência de 10-20 barras)
    - A combinação permite capturar tanto padrões locais quanto dependências temporais
    
    Conforme metodologia do trabalho (Seção 4.3 - Arquitetura dos Modelos).
    
    Parâmetros:
        n_steps: Número de barras históricas na janela temporal (padrão: 60)
        n_features: Número de features de entrada (padrão: 12)
        conv_filters: Número de filtros convolucionais (padrão: 64)
        conv_kernel_size: Tamanho do kernel convolucional (padrão: 2)
        pool_size: Tamanho do pooling para redução (padrão: 2)
        lstm_units: Número de unidades na camada LSTM (padrão: 50)
        dropout: Taxa de dropout para regularização (padrão: 0.2)
        learning_rate: Taxa de aprendizado do otimizador (padrão: 0.001)
    
    Retorna:
        Modelo Keras compilado e pronto para treinamento
    
    Exemplo:
        >>> model = criar_modelo_cnn_lstm(n_steps=60, n_features=12)
        >>> model.summary()
    """
    model = keras.Sequential([
        # Camada convolucional: extrai padrões locais
        layers.Conv1D(
            filters=conv_filters, 
            kernel_size=conv_kernel_size,
            activation='relu', 
            input_shape=(n_steps, n_features),
            name='conv1d_layer'
        ),
        # Pooling: reduz dimensionalidade e aumenta robustez
        layers.MaxPooling1D(
            pool_size=pool_size,
            name='maxpooling_layer'
        ),
        # LSTM: captura dependências temporais de longo prazo
        layers.LSTM(
            lstm_units, 
            dropout=dropout,
            name='lstm_layer'
        ),
        # Saída: classificação binária
        layers.Dense(
            1, 
            activation='sigmoid',
            name='output_layer'
        )
    ])
    
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
