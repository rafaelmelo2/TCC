"""Label Smoothing Loss para prevenir overconfidence."""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


def binary_crossentropy_with_label_smoothing(label_smoothing=0.1):
    """
    Binary crossentropy com label smoothing.
    
    Label smoothing suaviza os targets:
    - Ao invés de [0, 1], usa [epsilon, 1-epsilon]
    - Previne overconfidence e melhora generalização
    - Força modelo a não colapsar para uma classe
    
    Parâmetros:
        label_smoothing: Fator de suavização (0.0-0.5)
            - 0.0: sem suavização (binary crossentropy normal)
            - 0.1: suavização leve (recomendado)
            - 0.2: suavização moderada
    
    Exemplo:
        >>> loss = binary_crossentropy_with_label_smoothing(0.1)
        >>> model.compile(loss=loss, optimizer='adam')
    """
    def loss_fn(y_true, y_pred):
        # Aplicar label smoothing
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        
        # Binary crossentropy
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        loss = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
        
        return K.mean(loss)
    
    return loss_fn
