"""
Configurações globais compartilhadas em todo o projeto.

Centraliza apenas constantes que serão reutilizadas em múltiplos módulos.
Configurações específicas de um único módulo devem ficar no próprio módulo.
"""

from datetime import time

# ============================================================================
# ESTRUTURA DE DADOS
# ============================================================================

# Colunas obrigatórias para dados OHLCV (usado em load_data e validate_data)
COLUNAS_OBRIGATORIAS = ['data', 'abertura', 'maxima', 'minima', 'fechamento', 'volume_real']

# ============================================================================
# HORÁRIO DE PREGÃO B3
# ============================================================================

# Horário de pregão da B3 (usado em load_data e validate_data)
HORARIO_ABERTURA = time(10, 0)  # 10:00
HORARIO_FECHAMENTO = time(17, 0)  # 17:00

# ============================================================================
# CONFIGURAÇÕES DE FEATURES
# ============================================================================

# Períodos para indicadores técnicos (usado em feature_engineering e possivelmente em análise)
PERIODOS_EMA = [9, 21, 50]  # Médias Móveis Exponenciais
PERIODOS_RSI = [9, 21, 50]  # Relative Strength Index
PERIODO_BOLLINGER = 20  # Período para Bandas de Bollinger
DESVIOS_BOLLINGER = 2  # Número de desvios-padrão para Bandas de Bollinger
PERIODO_VOLATILIDADE = 20  # Janela para cálculo de volatilidade

# Threshold para banda morta (usado em feature_engineering e métricas)
THRESHOLD_BANDA_MORTA = 0.0005  # 0.05% - movimentos menores são considerados neutros

# ============================================================================
# CONFIGURAÇÕES DE WALK-FORWARD
# ============================================================================

# Tamanhos padrão para walk-forward validation (usado em validation e train)
# Baseado em barras de 15 minutos
BARRAS_POR_DIA = 26  # Aproximadamente (10h-17h = 7h = 28 barras, menos algumas)
TAMANHO_TREINO_BARRAS = 252 * 26  # ~1 ano de dados
TAMANHO_TESTE_BARRAS = 21 * 26    # ~1 mês de dados
EMBARGO_BARRAS = 1                # 1 barra de embargo entre treino e teste

# ============================================================================
# CONFIGURAÇÕES DE MODELOS
# ============================================================================

# Seed para reprodutibilidade (usado em train, modelos, e qualquer lugar que precise de aleatoriedade)
SEED = 42

# Janela temporal para modelos de deep learning (usado em train e modelos)
JANELA_TEMPORAL_STEPS = 60  # Número de barras históricas para prever próxima

# ============================================================================
# CONFIGURAÇÕES DE BACKTEST
# ============================================================================

# Custos de transação B3 (usado em backtest e possivelmente em análise de custos)
CUSTO_CORRETAGEM = 10.0           # R$ fixo por operação
CUSTO_TAXA_PROPORCIONAL = 0.0003  # 0.03% do volume negociado
CUSTO_SLIPPAGE = 0.0001           # 0.01% de slippage

# Capital inicial padrão para backtests (usado em backtest)
CAPITAL_INICIAL = 100000.0  # R$ 100.000

# ============================================================================
# CONFIGURAÇÕES GERAIS
# ============================================================================

# Intervalo padrão entre barras (usado em validate_data e possivelmente em outros lugares)
INTERVALO_BARRAS_MINUTOS = 15
