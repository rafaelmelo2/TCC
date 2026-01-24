"""
Configurações globais compartilhadas em todo o projeto.

Centraliza apenas constantes que serão reutilizadas em múltiplos módulos.
Configurações específicas de um único módulo devem ficar no próprio módulo.
"""

from datetime import time, datetime

# ============================================================================
# PERÍODO DOS DADOS
# ============================================================================

# IMPORTANTE: Período exato dos dados disponíveis
# Os arquivos estão nomeados como 20200101_20251231, mas os dados reais
# começam em 22/10/2020 (primeira barra disponível no MetaTrader 5)
# e vão até 22/10/2025 (5 anos completos a partir da primeira barra)
# 
# Documentado em: src/documentacao/periodo_dados.md

DATA_INICIO_DADOS = datetime(2020, 10, 22)  # Primeira barra disponível
DATA_FIM_DADOS = datetime(2025, 10, 22)      # 5 anos completos a partir do início

# Formato usado nos nomes de arquivos (mantido para compatibilidade)
# NOTA: Os arquivos mantêm o nome antigo (20200101_20251231) mas contêm
# dados apenas de 22/10/2020 até 22/10/2025
FORMATO_DATA_ARQUIVO_INICIO = "20200101"  # Nome do arquivo (não a data real)
FORMATO_DATA_ARQUIVO_FIM = "20251231"     # Nome do arquivo (não a data real)


def obter_nome_arquivo_dados(ticker: str, timeframe: str = "M15") -> str:
    """
    Retorna o nome padrão do arquivo de dados para um ticker.
    
    Parâmetros:
        ticker: Código do ativo (ex: 'PETR4', 'VALE3', 'ITUB4')
        timeframe: Timeframe dos dados (padrão: 'M15')
        
    Retorna:
        Nome do arquivo no formato: {TICKER}_{TIMEFRAME}_{DATA_INICIO}_{DATA_FIM}.csv
        
    Nota: O nome do arquivo usa as datas antigas (20200101_20251231) para
    compatibilidade, mas os dados reais são de 22/10/2020 até 22/10/2025.
    """
    return f"{ticker}_{timeframe}_{FORMATO_DATA_ARQUIVO_INICIO}_{FORMATO_DATA_ARQUIVO_FIM}.csv"


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
# IMPORTANTE: Para barras de 15 minutos, movimentos < 0.1% são considerados ruído
# Valor baseado em análise empírica de movimentos significativos intradiários
THRESHOLD_BANDA_MORTA = 0.001  # 0.1% - movimentos menores são considerados neutros

# ============================================================================
# CONFIGURAÇÕES DE WALK-FORWARD
# ============================================================================

# Tamanhos padrão para walk-forward validation (usado em validation e train)
# Baseado em barras de 15 minutos
# 
# METODOLOGIA: Configuração escolhida a priori baseada em literatura
# (Prado, 2018; Bergmeir & Benítez, 2012)
# 
# Justificativa:
# - Treino de 1 ano: Permite capturar padrões sazonais e ciclos anuais,
#   padrão comum em trabalhos de séries temporais financeiras
# - Teste de 1 mês: Permite avaliação frequente e detecção de mudanças
#   de regime de mercado ao longo do tempo
# - Embargo de 5 barras: Previne contaminação temporal mantendo viabilidade
#   computacional (~2 horas de separação entre treino e teste)
#
# IMPORTANTE: Esta configuração foi escolhida ANTES de executar experimentos
# para evitar overfitting de configuração (cherry-picking). Para análise de
# sensibilidade, ver: src/tests/testar_sensibilidade_walkforward.py

BARRAS_POR_DIA = 26  # Aproximadamente (10h-17h = 7h = 28 barras, menos algumas)

# CONFIGURAÇÃO PRINCIPAL (escolhida a priori)
TAMANHO_TREINO_BARRAS = 252 * 26  # ~1 ano de dados (6.552 barras)
TAMANHO_TESTE_BARRAS = 21 * 26    # ~1 mês de dados (546 barras)
EMBARGO_BARRAS = 5                # 5 barras de embargo (~2 horas)

# CONFIGURAÇÕES PARA ANÁLISE DE SENSIBILIDADE
# (Usadas em testar_sensibilidade_walkforward.py)
# Estas configurações serão testadas para validar robustez da escolha principal
CONFIGURACOES_SENSIBILIDADE = [
    {
        'nome': 'mais_permissivo',
        'treino': 252 * 26,  # 1 ano
        'teste': 21 * 26,     # 1 mês
        'embargo': 1,         # 1 barra (~4 minutos)
        'descricao': 'Configuração mais permissiva: embargo mínimo'
    },
    {
        'nome': 'principal',
        'treino': 252 * 26,  # 1 ano
        'teste': 21 * 26,     # 1 mês
        'embargo': 5,         # 5 barras (~2 horas) - CONFIGURAÇÃO PRINCIPAL
        'descricao': 'Configuração principal escolhida a priori'
    },
    {
        'nome': 'mais_conservador',
        'treino': 252 * 26,  # 1 ano
        'teste': 21 * 26,     # 1 mês
        'embargo': 10,        # 10 barras (~4 horas)
        'descricao': 'Configuração mais conservadora: embargo maior'
    },
    {
        'nome': 'mais_treino',
        'treino': 378 * 26,  # 1.5 anos
        'teste': 21 * 26,     # 1 mês
        'embargo': 5,         # 5 barras
        'descricao': 'Mais dados de treino: 1.5 anos'
    },
    {
        'nome': 'embargo_dia',
        'treino': 252 * 26,  # 1 ano
        'teste': 21 * 26,     # 1 mês
        'embargo': 26,        # 1 dia completo
        'descricao': 'Embargo de 1 dia completo (máxima separação)'
    }
]

# ============================================================================
# CONFIGURAÇÕES DE MODELOS
# ============================================================================

# Seed para reprodutibilidade (usado em train, modelos, e qualquer lugar que precise de aleatoriedade)
SEED = 42

# Janela temporal para modelos de deep learning (usado em train e modelos)
JANELA_TEMPORAL_STEPS = 60  # Número de barras históricas para prever próxima

# ============================================================================
# HIPERPARÂMETROS DE MODELOS (para otimização com Optuna)
# ============================================================================

# Espaços de busca para otimização bayesiana (Optuna)
# Conforme metodologia do TCC (Seção 4.4.2 - Seleção de Hiperparâmetros)
# 
# IMPORTANTE: Estes hiperparâmetros serão otimizados DENTRO de cada fold
# do walk-forward usando validação interna. A otimização é feita no conjunto
# de validação interno, não no conjunto de teste.

# Hiperparâmetros CNN-LSTM (Modelo Principal)
HIPERPARAMETROS_CNN_LSTM = {
    'conv_filters': [32, 64, 128],      # Número de filtros convolucionais
    'conv_kernel_size': [2, 3],         # Tamanho do kernel convolucional
    'pool_size': [2],                    # Tamanho do pooling (fixo)
    'lstm_units': [32, 50, 64],          # Número de unidades LSTM
    'dropout': [0.1, 0.2, 0.3],          # Taxa de dropout
    'learning_rate': [1e-4, 1e-3, 1e-2], # Taxa de aprendizado
    'batch_size': [16, 32, 64]           # Tamanho do batch
}

# Hiperparâmetros LSTM (Baseline)
HIPERPARAMETROS_LSTM = {
    'lstm_units': [32, 50, 64],          # Número de unidades LSTM
    'dropout': [0.1, 0.2, 0.3],          # Taxa de dropout
    'learning_rate': [1e-4, 1e-3, 1e-2], # Taxa de aprendizado
    'batch_size': [16, 32, 64]           # Tamanho do batch
}

# Valores padrão (usados quando não há otimização)
HIPERPARAMETROS_PADRAO_CNN_LSTM = {
    'conv_filters': 64,
    'conv_kernel_size': 2,
    'pool_size': 2,
    'lstm_units': 50,
    'dropout': 0.2,
    'learning_rate': 0.001
}

HIPERPARAMETROS_PADRAO_LSTM = {
    'lstm_units': 50,
    'dropout': 0.2,
    'learning_rate': 0.001
}

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
