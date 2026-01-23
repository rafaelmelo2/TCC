"""
Módulo de processamento de dados.

Inclui carregamento, validação e engenharia de features.
"""

from .load_data import carregar_dados, validar_estrutura_dados, obter_estatisticas_dados
from .validate_data import auditar_dados, gerar_relatorio_auditoria
from .feature_engineering import criar_features, calcular_retornos_logaritmicos

__all__ = [
    'carregar_dados',
    'validar_estrutura_dados',
    'obter_estatisticas_dados',
    'auditar_dados',
    'gerar_relatorio_auditoria',
    'criar_features',
    'calcular_retornos_logaritmicos'
]
