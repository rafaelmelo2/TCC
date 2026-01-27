"""Implementação de Focal Loss para problemas com classes desbalanceadas."""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementa Focal Loss para classificação binária.
    
    Focal Loss foca o treinamento em exemplos difíceis e mal classificados,
    reduzindo o peso de exemplos bem classificados. Isso previne que o modelo
    colapse para sempre prever a classe majoritária.
    
    Fórmula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    onde:
        - p_t: probabilidade prevista da classe verdadeira
        - gamma: focusing parameter (quanto maior, mais foco em difíceis)
        - alpha: balanceamento entre classes
    
    Parâmetros:
        gamma (float): Focusing parameter. Valores típicos: 2.0-5.0
            - gamma=0: equivalente a binary crossentropy
            - gamma=2: padrão (bom para maioria dos casos)
            - gamma=5: muito focado em exemplos difíceis
        
        alpha (float): Peso da classe positiva. Valores típicos: 0.25-0.75
            - alpha=0.25: mais peso para classe negativa
            - alpha=0.50: peso igual
            - alpha=0.75: mais peso para classe positiva
    
    Retorna:
        Função de loss compatível com Keras
    
    Referência:
        Lin et al. (2017): "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002
    
    Exemplo:
        >>> model.compile(
        ...     loss=focal_loss(gamma=2.0, alpha=0.25),
        ...     optimizer='adam',
        ...     metrics=['accuracy']
        ... )
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions para evitar log(0)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calcular p_t: probabilidade da classe verdadeira
        # Se y_true=1, p_t=y_pred; se y_true=0, p_t=(1-y_pred)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        
        # Calcular alpha_t: peso da classe
        # Se y_true=1, alpha_t=alpha; se y_true=0, alpha_t=(1-alpha)
        alpha_t = tf.where(K.equal(y_true, 1), alpha, 1 - alpha)
        
        # Calcular peso focal: (1 - p_t)^gamma
        # Exemplos bem classificados (p_t alto) têm peso baixo
        # Exemplos mal classificados (p_t baixo) têm peso alto
        focal_weight = alpha_t * K.pow((1 - p_t), gamma)
        
        # Calcular loss final
        loss = -focal_weight * K.log(p_t)
        
        return K.mean(loss)
    
    return focal_loss_fixed


def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    Alias para focal_loss() para compatibilidade.
    Mesma implementação, apenas nome diferente.
    """
    return focal_loss(gamma=gamma, alpha=alpha)


# Valores recomendados por tipo de problema
FOCAL_LOSS_CONFIGS = {
    'balanced': {
        'gamma': 2.0,
        'alpha': 0.5,
        'description': 'Classes balanceadas, foco moderado em difíceis'
    },
    'imbalanced_minor': {
        'gamma': 2.0,
        'alpha': 0.75,
        'description': 'Classe positiva minoritária, mais peso para ela'
    },
    'imbalanced_major': {
        'gamma': 2.0,
        'alpha': 0.25,
        'description': 'Classe positiva majoritária, menos peso para ela'
    },
    'very_hard': {
        'gamma': 5.0,
        'alpha': 0.5,
        'description': 'Exemplos muito difíceis, foco máximo neles'
    },
    'mild': {
        'gamma': 1.0,
        'alpha': 0.5,
        'description': 'Foco suave, próximo de binary crossentropy'
    }
}


def get_focal_loss(config_name='balanced'):
    """
    Retorna focal loss com configuração pré-definida.
    
    Parâmetros:
        config_name: Nome da configuração
            - 'balanced': Classes balanceadas (gamma=2, alpha=0.5)
            - 'imbalanced_minor': Classe positiva minoritária (gamma=2, alpha=0.75)
            - 'imbalanced_major': Classe positiva majoritária (gamma=2, alpha=0.25)
            - 'very_hard': Exemplos muito difíceis (gamma=5, alpha=0.5)
            - 'mild': Foco suave (gamma=1, alpha=0.5)
    
    Retorna:
        Função de loss
    
    Exemplo:
        >>> loss = get_focal_loss('imbalanced_minor')
        >>> model.compile(loss=loss, optimizer='adam')
    """
    if config_name not in FOCAL_LOSS_CONFIGS:
        raise ValueError(f"Configuração '{config_name}' não encontrada. "
                        f"Opções: {list(FOCAL_LOSS_CONFIGS.keys())}")
    
    config = FOCAL_LOSS_CONFIGS[config_name]
    print(f"[INFO] Usando Focal Loss: {config['description']}")
    print(f"       gamma={config['gamma']}, alpha={config['alpha']}")
    
    return focal_loss(gamma=config['gamma'], alpha=config['alpha'])
