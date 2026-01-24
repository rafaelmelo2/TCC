# Regras do Projeto - TCC Sprint Final (30 dias)

## ⚠️ MODO GUERRA - 22/01/2026 a 20/02/2026

Este arquivo contém as regras ABSOLUTAS para os próximos 30 dias de desenvolvimento intensivo do TCC.

## 🎯 Objetivo Central

Completar **TODA** a implementação experimental, análise de resultados e escrita do TCC2 em 30 dias, com defesa prevista para final de fevereiro de 2026.

## 🚫 Regras de Foco Absoluto

### Prioridade #1: TCC
- **TCC = ÚNICA PRIORIDADE** nos próximos 30 dias
- Nexarena → **PAUSADO**
- Bot da hamburgueria → **CONGELADO**
- Redes sociais → **BLOQUEADO** (usar app blocker)
- YouTube/entretenimento → **APENAS** após 22:00

### Rotina de Guerra (Segunda a Sexta)
```
05:00-08:00 → Academia (MANTER - saúde mental é crítica)
08:00-15:00 → Estágio HPE (obrigatório, não negociável)
15:30-16:00 → Almoço/descanso rápido
16:00-22:00 → TCC (6h puras, ZERO distração)
22:00-23:00 → Revisão do dia + planning do próximo
23:00-05:00 → SONO (6h mínimo, NÃO NEGOCIÁVEL)
```

### Fins de Semana (Sábado e Domingo)
```
08:00-09:00 → Café da manhã
09:00-13:00 → TCC (4h)
13:00-14:00 → Almoço
14:00-18:00 → TCC (4h)
18:00-20:00 → Descanso ativo (caminhada, família)
20:00-22:00 → TCC (2h finais)
Total: 10h/dia nos fins de semana
```

## 📊 Métricas de Progresso

### Tracking Diário (OBRIGATÓRIO)
Ao final de cada dia, atualizar `PROGRESSO.md` com:
- [ ] Tarefas completadas
- [ ] Tarefas pendentes
- [ ] Bloqueios encontrados
- [ ] Tempo efetivo de trabalho
- [ ] Nível de energia (1-10)

### Red Flags (Alertas Críticos)
🚨 Se qualquer um ocorrer, **PARE E REAVALIE**:
- Mais de 2 dias consecutivos sem código novo
- Acurácia dos modelos < 50% (indica problema fundamental)
- GPU travando/sem memória (otimizar ANTES de continuar)
- Walk-forward levando >12h para rodar (paralelizar)
- Sono < 5h por mais de 2 noites

## 🏗️ Estrutura de Código - Padrões Obrigatórios

### Organização de Pastas (SEGUIR ESTRITAMENTE)
```
codigo/pipeline/
├── data/
│   ├── raw/              # Dados brutos (CSV originais)
│   ├── processed/        # Dados limpos e validados
│   └── features/         # Dados com indicadores técnicos
├── src/
│   ├── data_processing/
│   │   ├── load_data.py
│   │   ├── validate_data.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── baselines.py       # Naive, Drift
│   │   ├── arima_model.py
│   │   ├── prophet_model.py
│   │   ├── lstm_model.py
│   │   └── cnn_lstm_model.py  # Modelo proposto
│   ├── utils/
│   │   ├── validation.py      # WalkForwardValidator
│   │   ├── metrics.py         # Todas as métricas
│   │   └── backtesting.py     # SimpleBacktest
│   ├── train.py              # Script principal de treino
│   ├── evaluate.py           # Avaliação de modelos
│   └── backtest.py           # Backtest com custos
├── results/
│   ├── models/               # Checkpoints (.keras)
│   ├── metrics/              # CSVs com resultados
│   ├── plots/                # Gráficos e visualizações
│   └── logs/                 # Logs de experimentos
├── notebooks/
│   └── exploratory.ipynb     # Análise exploratória
└── tests/
    └── test_*.py             # Testes unitários
```

### Nomenclatura de Arquivos
- Modelos salvos: `{modelo}_{ativo}_{fold}_{timestamp}.keras`
  - Exemplo: `cnn_lstm_PETR4_fold_03_20260125_1430.keras`
- Resultados: `{modelo}_{metrica}_{data}.csv`
  - Exemplo: `cnn_lstm_metrics_20260125.csv`
- Plots: `{tipo}_{modelo}_{ativo}.png`
  - Exemplo: `accuracy_evolution_cnn_lstm_VALE3.png`

### Convenções de Código
```python
# SEMPRE usar estas convenções:
# - Funções: snake_case
# - Classes: PascalCase
# - Constantes: UPPER_SNAKE_CASE
# - Variáveis: snake_case

# Exemplo:
N_STEPS = 60  # Constante
class WalkForwardValidator:  # Classe
    def __init__(self):
        self.train_size = 252 * 26  # Variável
        
    def get_folds(self):  # Função
        pass
```

### Docstrings OBRIGATÓRIAS
```python
def create_features(df: pd.DataFrame, indicators: list) -> pd.DataFrame:
    """
    Cria indicadores técnicos para séries temporais intradiárias.
    
    Conforme Seção 4.2 do TCC (Engenharia de Atributos).
    
    Parâmetros:
        df: DataFrame com OHLCV (colunas: open, high, low, close, volume)
        indicators: Lista de indicadores ['ema', 'rsi', 'bollinger']
        
    Retorna:
        DataFrame com colunas originais + indicadores calculados
        
    Exceções:
        ValueError: Se df não contiver colunas OHLCV obrigatórias
        
    Exemplo:
        >>> df = pd.read_csv('PETR4_15min.csv')
        >>> df_features = create_features(df, ['ema', 'rsi'])
    """
    # Implementação...
    pass
```

## 🔬 Validação e Testes

### Checklist Antes de Cada Commit
- [ ] Código roda sem erros
- [ ] Docstrings estão completas
- [ ] Logs informativos estão presentes (`[OK]`, `[!]`, `[ERRO]`)
- [ ] Seeds estão fixadas (42)
- [ ] Nenhum `shuffle=True` em dados temporais
- [ ] Normalização aplicada APENAS no treino

### Prevenção de Data Leakage (CRÍTICO)
```python
# ❌ ERRADO - Normaliza tudo junto
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)

# ✅ CORRETO - Normaliza separadamente
for fold in folds:
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(fold['train'])  # Fit no treino
    test_scaled = scaler.transform(fold['test'])        # Transform no teste
```

### Walk-Forward OBRIGATÓRIO
```python
# ❌ NUNCA FAZER ISSO
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True)  # VIOLA ORDEM TEMPORAL!

# ✅ SEMPRE FAZER ISSO
from utils.validation import WalkForwardValidator
validator = WalkForwardValidator(
    data=df,
    train_size=252*26,  # ~1 ano
    test_size=21*26,    # ~1 mês
    embargo=1           # 1 barra de embargo
)
folds = validator.get_folds()
```

## 💾 Reprodutibilidade

### Seeds Fixadas
```python
# Início de TODOS os scripts
import random
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

### Versionamento de Dependências
Manter atualizado `requirements.txt`:
```
pandas==2.1.4
numpy==1.24.3
tensorflow-gpu==2.13.0
scikit-learn==1.3.2
optuna==3.4.0
statsmodels==0.14.0
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0
```

### Logging de Experimentos
```python
# SEMPRE logar configurações
import json
from datetime import datetime

config = {
    'model': 'cnn_lstm',
    'asset': 'PETR4',
    'fold': 3,
    'hyperparameters': {
        'filters': [64, 32],
        'lstm_units': [64, 32],
        'dropout': 0.2,
        'learning_rate': 0.001
    },
    'timestamp': datetime.now().isoformat()
}

with open(f'results/logs/config_{config["timestamp"]}.json', 'w') as f:
    json.dump(config, f, indent=2)
```

## 📈 Métricas - Calcular TODAS

### Preditivas
```python
from utils.metrics import compute_all_metrics

metrics = compute_all_metrics(
    y_true=y_test,
    y_pred=predictions,
    y_prob=probabilities
)

# Deve retornar:
# {
#     'accuracy': float,           # Acurácia direcional
#     'balanced_accuracy': float,  # Com ponderação de classes
#     'f1_score': float,
#     'mcc': float,               # Matthews Correlation Coef
#     'brier_score': float,       # Qualidade probabilística
#     'log_loss': float,
#     'auc_pr': float,            # Área sob Precision-Recall
#     'ece': float                # Expected Calibration Error
# }
```

### Trading (Pós-Custos)
```python
from utils.backtesting import SimpleBacktest

backtest = SimpleBacktest(
    costs={
        'corretagem': 10.0,     # R$ fixo por operação
        'taxa': 0.0003,         # 0.03% do volume
        'slippage': 0.0001      # 0.01% de slippage
    }
)

results = backtest.run(df=test_data, signals=predictions)

# Deve retornar:
# {
#     'final_value': float,
#     'return_pct': float,
#     'sharpe_ratio': float,
#     'max_drawdown': float,
#     'profit_factor': float,
#     'turnover': float,
#     'num_trades': int
# }
```

## 🚨 Alertas e Debugging

### Problemas Comuns e Soluções

#### 1. Modelo não converge
```python
# Sintomas: Loss não diminui, fica em ~0.69 (log(2))
# Soluções:
# - Reduzir learning rate: 0.001 → 0.0001
# - Adicionar BatchNormalization
# - Simplificar arquitetura (menos camadas)
# - Verificar se labels estão balanceadas
```

#### 2. Acurácia = 50% (chute aleatório)
```python
# Sintomas: Modelo sempre prevê mesma classe
# Soluções:
# - Verificar balanceamento (usar class_weight)
# - Aumentar capacidade do modelo
# - Revisar engenharia de features
# - Checar se há data leakage
```

#### 3. GPU sem memória
```python
# Sintomas: CUDA out of memory
# Soluções:
# - Reduzir batch_size: 64 → 32 → 16
# - Reduzir tamanho das janelas: 60 → 30
# - Usar gradient_checkpointing
# - Limpar cache: tf.keras.backend.clear_session()
```

#### 4. Walk-forward muito lento
```python
# Sintomas: Cada fold leva >2h
# Soluções:
# - Paralelizar folds (joblib, multiprocessing)
# - Reduzir número de trials do Optuna: 50 → 20
# - Usar early stopping agressivo
# - Cache de features processadas
```

## 📝 Comunicação com Orientador

### Reuniões Semanais
- **Quando**: Toda sexta às 16:00
- **Duração**: 30min
- **Formato**: 
  1. Progresso da semana (5min)
  2. Resultados preliminares (10min)
  3. Problemas encontrados (10min)
  4. Planejamento próxima semana (5min)

### E-mails
- **Subject**: `[TCC] - Semana X - {Tópico}`
- **Frequência**: Mínimo 1x por semana
- **Conteúdo**: 
  - Resumo executivo (3 linhas)
  - Progresso em bullet points
  - Próximos passos
  - Anexar gráficos/tabelas relevantes

## 🎓 Escrita do TCC2

### Estrutura do Capítulo de Resultados
```markdown
# 5. RESULTADOS E DISCUSSÃO

## 5.1 Descrição dos Dados
- Estatísticas descritivas (Tabela)
- Gráficos de série temporal
- Testes de estacionariedade (ADF)

## 5.2 Desempenho Preditivo
- Tabela consolidada: Modelo × Métrica × Ativo
- Gráficos de acurácia por fold
- Curvas de calibração
- Teste Diebold-Mariano

## 5.3 Desempenho Operacional
- Backtests: Retorno, Sharpe, Drawdown
- Curvas de equity
- Análise de turnover

## 5.4 Análise de Robustez
- Resultados por regime de volatilidade
- Sensibilidade a custos
- Comparação entre ativos

## 5.5 Discussão
- Por que CNN-LSTM superou (ou não) baselines?
- Limitações do estudo
- Implicações práticas
```

### Tabelas e Figuras
- **Numeração**: Sequencial dentro de cada capítulo
  - Exemplo: Tabela 5.1, Figura 5.2
- **Legendas**: SEMPRE abaixo (tabelas) ou abaixo (figuras)
- **Fonte**: Times New Roman 10pt para legendas
- **Referência**: Todas devem ser citadas no texto

### Citações (ABNT)
```
# No texto:
Conforme Vaswani et al. (2017), a arquitetura Transformer...
...mecanismo de atenção (VASWANI et al., 2017).

# Na referência:
VASWANI, A. et al. Attention is all you need. In: ADVANCES IN 
NEURAL INFORMATION PROCESSING SYSTEMS. 2017. p. 5998-6008.
```

## ⏱️ Milestones Críticos

### Semana 1 (22-28 Jan) - FUNDAÇÃO
- ✅ Dados auditados
- ✅ Features criadas
- ✅ Baselines rodando
- ✅ Walk-forward implementado

### Semana 2 (29 Jan - 04 Fev) - LSTM
- ✅ LSTM puro treinado
- ✅ Optuna rodando
- ✅ Primeiros resultados

### Semana 3 (05-11 Fev) - CNN-LSTM
- ✅ CNN-LSTM treinado
- ✅ Comparação com baselines
- ✅ Backtests completos

### Semana 4 (12-18 Fev) - ANÁLISES
- ✅ Testes estatísticos
- ✅ Regimes de volatilidade
- ✅ Sensibilidades

### Semana 5 (19-20 Fev) - ESCRITA
- ✅ Resultados redigidos
- ✅ Discussão completa
- ✅ Revisão final

## 🚀 Mantra do Projeto

```
"Feito é melhor que perfeito.
Reprodutível é melhor que otimizado.
Documentado é melhor que elegante.
Entregue é melhor que em progresso."
```

---

**Data de Início**: 22/01/2026  
**Data de Entrega**: 20/02/2026  
**Dias Restantes**: 30 dias  

**FOCO TOTAL. ZERO DISTRAÇÕES. VAMOS TERMINAR ISSO! 💪🔥**
