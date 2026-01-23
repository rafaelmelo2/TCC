# 📅 CRONOGRAMA TCC2 - 30 DIAS DE GUERRA

**Período**: 22 de Janeiro a 20 de Fevereiro de 2026  
**Objetivo**: Completar implementação, análises e escrita do TCC2  
**Status**: 🔴 EM ANDAMENTO

---

## 📊 Visão Geral

| Semana | Período | Foco Principal | Entregáveis Principais |
|--------|---------|----------------|------------------------|
| **1** | 22-28 Jan | Preparação & Baselines | Dados validados, Features, Naive/ARIMA rodando |
| **2** | 29 Jan-04 Fev | LSTM Puro | Modelo LSTM otimizado, Primeiros resultados |
| **3** | 05-11 Fev | CNN-LSTM Híbrido | Modelo proposto completo, Comparações |
| **4** | 12-18 Fev | Análises & Testes | Testes estatísticos, Robustez, Sensibilidade |
| **5** | 19-20 Fev | Finalização | Escrita completa, Revisão final, Slides |

---

## 🗓️ SEMANA 1: PREPARAÇÃO DO CAMPO DE BATALHA (22-28 Jan)

### 🎯 Objetivo da Semana
Garantir que a infraestrutura está sólida: dados limpos, features criadas, baselines funcionando e walk-forward implementado.

---

### **Quarta-feira, 22/01 (DIA 1) - HOJE** 🔥
**Tema**: Auditoria Técnica Completa

#### Bloco 1 (16:00-18:00): Organização de Dados
- [x] Verificar estrutura dos dados brutos em `data/raw/`
- [x] Validar período de cobertura (Jan/2020 - Jul/2025)
- [x] Checar missing values e gaps
- [x] Confirmar ajustes por splits/dividendos
- [x] Validar timestamps (timezone, horário de pregão 10h-17h)

```python
# Script: src/data_processing/validate_data.py
for ativo in ['PETR4', 'VALE3', 'ITUB4']:
    df = pd.read_csv(f'data/raw/{ativo}_M15_20200101_20251022.csv')
    print(f"\n{ativo}:")
    print(f"  Shape: {df.shape}")
    print(f"  Período: {df['time'].min()} até {df['time'].max()}")
    print(f"  Missing: {df.isnull().sum().sum()}")
    print(f"  Colunas: {df.columns.tolist()}")
```

**Entregável**: Relatório de auditoria (`AUDITORIA_DADOS.md`)

#### Bloco 2 (18:00-20:00): Ambiente de Desenvolvimento
- [ ] Criar ambiente conda: `conda create -n tcc python=3.10`
- [ ] Instalar dependências essenciais
- [ ] Testar GPU (NVIDIA 1660 Super)
- [ ] Configurar TensorFlow-GPU

```bash
# Comandos
conda activate tcc
pip install pandas numpy scikit-learn
pip install tensorflow-gpu==2.13.0
pip install optuna statsmodels matplotlib seaborn plotly
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Entregável**: Ambiente funcionando + screenshot da GPU detectada

#### Bloco 3 (20:00-22:00): Estrutura de Código
- [x] Criar estrutura de pastas conforme `.cursor/rules.md`
- [x] Organizar dados em `raw/`, `processed/`, `features/`
- [x] Criar esqueletos dos módulos principais
- [ ] Inicializar Git (se ainda não feito)

**Entregável**: Estrutura completa + README.md atualizado

---

### **Quinta-feira, 23/01 (DIA 2)**
**Tema**: Engenharia de Features

#### Bloco 1 (16:00-18:00): Implementação de Indicadores Técnicos
- [x] Criar `src/data_processing/feature_engineering.py`
- [x] Implementar retornos logarítmicos
- [x] Implementar MME (9, 21, 50 períodos)
- [x] Implementar RSI (9, 21, 50 períodos)

```python
# feature_engineering.py
def create_features(df):
    # Retornos log
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # MME
    for period in [9, 21, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df
```

**Entregável**: Script de features testado em PETR4

#### Bloco 2 (18:00-20:00): Mais Indicadores
- [x] Implementar Bandas de Bollinger (20, 2σ)
- [x] Implementar volatilidade realizada (janela 20)
- [x] Criar labels com banda morta (threshold=0.0005)

```python
# Continuação feature_engineering.py
# Bandas de Bollinger
sma_20 = df['close'].rolling(window=20).mean()
std_20 = df['close'].rolling(window=20).std()
df['bb_upper'] = sma_20 + (std_20 * 2)
df['bb_lower'] = sma_20 - (std_20 * 2)
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20

# Volatilidade
df['volatility'] = df['returns'].rolling(window=20).std()

# Target com banda morta
df['next_return'] = df['returns'].shift(-1)
threshold = 0.0005
df['target'] = 0  # Neutro
df.loc[df['next_return'] > threshold, 'target'] = 1   # Alta
df.loc[df['next_return'] < -threshold, 'target'] = -1  # Baixa
```

**Entregável**: Dataset completo com features salvo em `data/features/`

#### Bloco 3 (20:00-22:00): Análise Exploratória
- [ ] Criar notebook `notebooks/exploratory.ipynb`
- [ ] Gráficos de série temporal
- [ ] Distribuição de retornos
- [ ] Correlação entre features
- [ ] Estatísticas descritivas

**Entregável**: Notebook com análise exploratória completa

---

### **Sexta-feira, 24/01 (DIA 3)**
**Tema**: Baselines Naive e ARIMA

#### Bloco 1 (16:00-18:00): Baseline Naive
- [x] Implementar `src/models/baselines.py`
- [x] Classe `NaiveBaseline` (repete último movimento)
- [x] Classe `DriftBaseline` (tendência linear)
- [x] Testar em um ativo

```python
# baselines.py
class NaiveBaseline:
    """Assume próximo movimento = último movimento"""
    def __init__(self):
        self.name = 'Naive'
    
    def predict(self, series):
        # Retorna direção do último movimento
        last_return = series.iloc[-1]
        if last_return > 0.0005:
            return 1  # Alta
        elif last_return < -0.0005:
            return -1  # Baixa
        else:
            return 0  # Neutro
```

**Entregável**: Baselines naive rodando

#### Bloco 2 (18:00-20:00): Baseline ARIMA
- [x] Implementar classe `ARIMABaseline`
- [x] Grid search para (p,d,q) otimizado por AIC
- [ ] Treinar em dados históricos (requer statsmodels instalado)

```python
from statsmodels.tsa.arima.model import ARIMA

class ARIMABaseline:
    def __init__(self):
        self.model = None
        self.best_order = None
    
    def fit(self, train_data):
        best_aic = np.inf
        # Grid search simples
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(train_data, order=(p,d,q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            self.model = fitted
                            self.best_order = (p,d,q)
                    except:
                        continue
    
    def predict(self, steps=1):
        return self.model.forecast(steps=steps)
```

**Entregável**: ARIMA funcionando com ordem otimizada

#### Bloco 3 (20:00-22:00): Primeiras Métricas
- [x] Implementar `src/utils/metrics.py`
- [x] Calcular acurácia direcional
- [x] Calcular RMSE
- [ ] Comparar Naive vs ARIMA (aguardando walk-forward)

```python
# metrics.py
def compute_directional_accuracy(y_true, y_pred, dead_band=0.0005):
    """
    Acurácia com banda morta.
    Ignora movimentos neutros (|return| < threshold).
    """
    mask = np.abs(y_true) > dead_band
    y_true_filtered = (y_true[mask] > 0).astype(int)
    y_pred_filtered = (y_pred[mask] > 0).astype(int)
    
    return accuracy_score(y_true_filtered, y_pred_filtered)
```

**Entregável**: Primeiras métricas calculadas e salvas

---

### **Sábado, 25/01 (DIA 4)**
**Tema**: Walk-Forward Validation

#### Manhã (09:00-13:00): Implementação Walk-Forward
- [ ] Criar `src/utils/validation.py`
- [ ] Classe `WalkForwardValidator`
- [ ] Definir tamanhos de janelas (train=1 ano, test=1 mês)
- [ ] Implementar embargo temporal

```python
# validation.py
class WalkForwardValidator:
    def __init__(self, data, train_size=252*26, test_size=21*26, embargo=1):
        """
        train_size: ~1 ano de barras de 15min
        test_size: ~1 mês  
        embargo: 1 barra entre train/test
        """
        self.data = data
        self.train_size = train_size
        self.test_size = test_size
        self.embargo = embargo
        
    def get_folds(self):
        folds = []
        n = len(self.data)
        start = 0
        
        while start + self.train_size + self.test_size <= n:
            train_end = start + self.train_size
            test_start = train_end + self.embargo
            test_end = test_start + self.test_size
            
            folds.append({
                'train': self.data.iloc[start:train_end],
                'test': self.data.iloc[test_start:test_end],
                'fold_id': len(folds)
            })
            
            # Avança 1 mês
            start += self.test_size
            
        return folds
```

**Entregável**: Walk-forward funcionando

#### Tarde (14:00-18:00): Testar Walk-Forward nos Baselines
- [ ] Rodar Naive em walk-forward completo
- [ ] Rodar ARIMA em walk-forward completo
- [ ] Salvar métricas por fold
- [ ] Gerar gráfico de acurácia ao longo do tempo

**Entregável**: Resultados de baselines por fold em CSV

#### Noite (20:00-22:00): Análise Preliminar
- [ ] Analisar evolução da acurácia
- [ ] Identificar períodos problemáticos
- [ ] Documentar padrões observados

**Entregável**: Notebook com análise de baselines

---

### **Domingo, 26/01 (DIA 5)**
**Tema**: Refinamento e Documentação

#### Manhã (09:00-13:00): Prophet Baseline
- [ ] Implementar `ProphetBaseline`
- [ ] Adaptar para dados intradiários
- [ ] Testar sazonalidades (diária)

```python
from fbprophet import Prophet

class ProphetBaseline:
    def __init__(self):
        self.model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=False,
            yearly_seasonality=False
        )
    
    def fit(self, train_data):
        df = train_data[['timestamp', 'close']].rename(
            columns={'timestamp': 'ds', 'close': 'y'}
        )
        self.model.fit(df)
    
    def predict(self, periods):
        future = self.model.make_future_dataframe(periods=periods, freq='15min')
        forecast = self.model.predict(future)
        return forecast['yhat'].values[-periods:]
```

**Entregável**: Prophet funcionando

#### Tarde (14:00-18:00): Consolidação de Resultados
- [ ] Tabela comparativa: Naive vs ARIMA vs Prophet
- [ ] Gráficos de comparação
- [ ] Análise de erros

**Entregável**: Relatório de baselines (`BASELINES_REPORT.md`)

#### Noite (20:00-22:00): Planejamento da Semana 2
- [ ] Revisar arquitetura LSTM
- [ ] Listar hiperparâmetros para Optuna
- [ ] Preparar ambiente de treino

**Entregável**: Checklist da Semana 2

---

### **Segunda-feira, 27/01 (DIA 6)**
**Tema**: Preparação para Deep Learning

#### Bloco 1 (16:00-18:00): Preparação de Dados para DL
- [ ] Criar sequências de janelas temporais (60 barras)
- [ ] Implementar `create_sequences()`
- [ ] Normalização Min-Max dentro de cada fold

```python
def create_sequences(data, n_steps=60):
    """
    Cria sequências de janelas para LSTM.
    
    Parâmetros:
        data: DataFrame com features
        n_steps: Tamanho da janela temporal
        
    Retorna:
        X: (n_samples, n_steps, n_features)
        y: (n_samples,)
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data.iloc[i:i+n_steps].values)
        y.append(data.iloc[i+n_steps]['target'])
    
    return np.array(X), np.array(y)
```

**Entregável**: Pipeline de dados para LSTM

#### Bloco 2 (18:00-20:00): Arquitetura LSTM Básica
- [ ] Criar `src/models/lstm_model.py`
- [ ] Implementar arquitetura básica (2 camadas)
- [ ] Testar compilação e forward pass

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm(input_shape, units=[64, 32], dropout=0.2, lr=0.001):
    model = Sequential([
        LSTM(units[0], return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units[1], return_sequences=False),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

**Entregável**: LSTM compilando sem erros

#### Bloco 3 (20:00-22:00): Setup Optuna
- [ ] Criar script de otimização `src/train_optuna.py`
- [ ] Definir espaço de busca de hiperparâmetros
- [ ] Configurar study

```python
import optuna

def objective(trial):
    # Hiperparâmetros
    units_1 = trial.suggest_int('units_1', 32, 128, step=32)
    units_2 = trial.suggest_int('units_2', 16, 64, step=16)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Treinar modelo
    model = build_lstm(
        input_shape=(n_steps, n_features),
        units=[units_1, units_2],
        dropout=dropout,
        lr=lr
    )
    
    # ... código de treino ...
    
    return val_accuracy

# Criar study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

**Entregável**: Optuna configurado e testado

---

### **Terça-feira, 28/01 (DIA 7)**
**Tema**: Fechamento da Semana 1

#### Bloco 1 (16:00-18:00): Testes Finais
- [ ] Rodar pipeline completo end-to-end
- [ ] Verificar reprodutibilidade (seeds)
- [ ] Checar logs e outputs

#### Bloco 2 (18:00-20:00): Documentação
- [ ] Atualizar README.md
- [ ] Documentar decisões tomadas
- [ ] Preparar apresentação para orientador

#### Bloco 3 (20:00-22:00): Revisão e Planning
- [ ] Revisar progresso da semana
- [ ] Atualizar `PROGRESSO.md`
- [ ] Planejar Semana 2 em detalhes

**Entregável**: 
- ✅ Dados validados e limpos
- ✅ Features criadas e testadas
- ✅ Baselines (Naive, ARIMA, Prophet) funcionando
- ✅ Walk-forward implementado
- ✅ Ambiente de DL pronto

---

## 🗓️ SEMANA 2: LSTM PURO (29 Jan - 04 Fev)

### 🎯 Objetivo da Semana
Implementar, treinar e otimizar o modelo LSTM puro, gerando os primeiros resultados de deep learning.

---

### **Quarta-feira, 29/01 (DIA 8)**
**Tema**: Primeiro Treino LSTM

#### Bloco 1 (16:00-18:00): Script de Treino
- [ ] Criar `src/train.py` completo
- [ ] Implementar callbacks (EarlyStopping, ModelCheckpoint)
- [ ] Testar em 1 fold

```python
# train.py
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath='results/models/lstm_{fold}_best.keras',
    monitor='val_loss',
    save_best_only=True
)

# Treino
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)
```

**Entregável**: Primeiro modelo LSTM treinado

#### Bloco 2 (18:00-20:00): Análise do Treino
- [ ] Plotar curvas de loss
- [ ] Verificar overfitting
- [ ] Ajustar se necessário

#### Bloco 3 (20:00-22:00): Walk-Forward com LSTM
- [ ] Adaptar para walk-forward
- [ ] Treinar em múltiplos folds
- [ ] Salvar checkpoints

**Entregável**: LSTM rodando em walk-forward

---

### **Quinta-feira, 30/01 (DIA 9)**
**Tema**: Otimização Bayesiana

#### Bloco 1 (16:00-20:00): Optuna Rodando
- [ ] Iniciar otimização com Optuna (50 trials)
- [ ] **Deixar rodando overnight**
- [ ] Monitorar progresso

```bash
# Comando para rodar
nohup python src/train_optuna.py --asset PETR4 --n_trials 50 > logs/optuna_petr4.log 2>&1 &
```

#### Bloco 2 (20:00-22:00): Monitoramento
- [ ] Verificar trials completados
- [ ] Analisar melhores hiperparâmetros até agora
- [ ] Ajustar espaço de busca se necessário

**Entregável**: Optuna rodando (esperado: terminar no dia seguinte)

---

### **Sexta-feira, 31/01 (DIA 10)**
**Tema**: Análise Optuna

#### Bloco 1 (16:00-18:00): Resultados Optuna
- [ ] Coletar melhores hiperparâmetros
- [ ] Salvar em `results/models/best_hyperparams_lstm.json`
- [ ] Visualizar história de otimização

```python
# Análise Optuna
print("Melhores hiperparâmetros:", study.best_params)
print("Melhor acurácia:", study.best_value)

# Salvar
with open('results/models/best_hyperparams_lstm.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)

# Plotar
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

**Entregável**: Hiperparâmetros ótimos encontrados

#### Bloco 2 (18:00-20:00): Retreinamento
- [ ] Retreinar com melhores hiperparâmetros
- [ ] Rodar em todos os folds do walk-forward
- [ ] Salvar modelos finais

#### Bloco 3 (20:00-22:00): Métricas Completas
- [ ] Calcular todas as métricas (accuracy, F1, MCC, Brier, etc.)
- [ ] Comparar com baselines
- [ ] Gerar tabelas e gráficos

**Entregável**: LSTM otimizado com resultados completos

---

### **Sábado, 01/02 (DIA 11)**
**Tema**: Análise de Resultados LSTM

#### Manhã (09:00-13:00): Análise Profunda
- [ ] Curvas de calibração
- [ ] Análise de erros
- [ ] Casos onde LSTM falha
- [ ] Comparação fold-a-fold com baselines

**Entregável**: Notebook de análise LSTM

#### Tarde (14:00-18:00): Testes Adicionais
- [ ] Testar em VALE3 e ITUB4
- [ ] Verificar generalização
- [ ] Comparar performance entre ativos

#### Noite (20:00-22:00): Documentação
- [ ] Atualizar `PROGRESSO.md`
- [ ] Documentar arquitetura escolhida
- [ ] Preparar relatório semanal para orientador

**Entregável**: Relatório LSTM completo

---

### **Domingo, 02/02 (DIA 12)**
**Tema**: Preparação CNN-LSTM

#### Manhã (09:00-13:00): Estudo de Arquitetura
- [ ] Revisar arquitetura CNN-LSTM do TCC
- [ ] Estudar exemplos de Conv1D para séries temporais
- [ ] Planejar implementação

#### Tarde (14:00-18:00): Esqueleto CNN-LSTM
- [ ] Criar `src/models/cnn_lstm_model.py`
- [ ] Implementar arquitetura básica
- [ ] Testar compilação

```python
from tensorflow.keras.layers import Conv1D, MaxPooling1D

def build_cnn_lstm(input_shape, filters=[64, 32], kernel_size=3,
                   lstm_units=[64, 32], dropout=0.2, lr=0.001):
    model = Sequential([
        # Camadas Conv1D
        Conv1D(filters[0], kernel_size, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters[1], kernel_size, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Camadas LSTM
        LSTM(lstm_units[0], return_sequences=True),
        Dropout(dropout),
        LSTM(lstm_units[1], return_sequences=False),
        Dropout(dropout),
        
        # Classificador
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

**Entregável**: CNN-LSTM compilando

#### Noite (20:00-22:00): Planejamento Semana 3
- [ ] Revisar cronograma
- [ ] Listar tarefas da Semana 3
- [ ] Preparar ambiente

---

### **Segunda-feira, 03/02 (DIA 13)**
**Tema**: Primeiros Testes CNN-LSTM

#### Bloco 1 (16:00-18:00): Treino Inicial
- [ ] Treinar CNN-LSTM em 1 fold
- [ ] Comparar com LSTM puro
- [ ] Analisar curvas de aprendizado

#### Bloco 2 (18:00-20:00): Ajustes
- [ ] Ajustar arquitetura se necessário
- [ ] Testar diferentes kernel_size
- [ ] Verificar capacidade do modelo

#### Bloco 3 (20:00-22:00): Preparar Optuna CNN-LSTM
- [ ] Adaptar script de otimização
- [ ] Definir espaço de busca (filters, kernel, lstm_units)
- [ ] Configurar study

**Entregável**: CNN-LSTM treinando, Optuna preparado

---

### **Terça-feira, 04/02 (DIA 14)**
**Tema**: Fechamento Semana 2

#### Bloco 1 (16:00-18:00): Iniciar Optuna CNN-LSTM
- [ ] Iniciar otimização (deixar rodando overnight)
- [ ] Configurar monitoramento

```bash
nohup python src/train_optuna_cnn_lstm.py --asset PETR4 --n_trials 50 > logs/optuna_cnn_lstm.log 2>&1 &
```

#### Bloco 2 (18:00-20:00): Revisão Semanal
- [ ] Consolidar resultados LSTM
- [ ] Preparar apresentação para orientador
- [ ] Reunião com orientador (sexta 16:00)

#### Bloco 3 (20:00-22:00): Atualização
- [ ] Atualizar `PROGRESSO.md`
- [ ] Planejar Semana 3
- [ ] Revisar milestones

**Entregável**:
- ✅ LSTM otimizado e avaliado
- ✅ Comparação com baselines
- ✅ CNN-LSTM implementado
- ✅ Optuna CNN-LSTM rodando

---

## 🗓️ SEMANA 3: CNN-LSTM HÍBRIDO (05-11 Fev)

### 🎯 Objetivo da Semana
Completar otimização, treino e avaliação do modelo CNN-LSTM híbrido (modelo proposto). Realizar backtests iniciais.

---

### **Quarta-feira, 05/02 (DIA 15)**
**Tema**: Análise Optuna CNN-LSTM

#### Bloco 1 (16:00-18:00): Coleta de Resultados
- [ ] Analisar trials do Optuna
- [ ] Identificar melhores hiperparâmetros
- [ ] Salvar configuração ótima

#### Bloco 2 (18:00-20:00): Retreinamento
- [ ] Retreinar com hiperparâmetros ótimos
- [ ] Validar em fold de validação
- [ ] Ajustar se necessário

#### Bloco 3 (20:00-22:00): Walk-Forward Completo
- [ ] Iniciar walk-forward com CNN-LSTM otimizado
- [ ] Treinar em todos os folds
- [ ] Salvar checkpoints

**Entregável**: CNN-LSTM otimizado rodando

---

### **Quinta-feira, 06/02 (DIA 16)**
**Tema**: Avaliação Completa

#### Bloco 1 (16:00-18:00): Métricas Preditivas
- [ ] Calcular todas as métricas por fold
- [ ] Agregar resultados
- [ ] Comparar: Naive vs ARIMA vs Prophet vs LSTM vs CNN-LSTM

```python
# Tabela comparativa
models = ['Naive', 'ARIMA', 'Prophet', 'LSTM', 'CNN-LSTM']
metrics = ['Accuracy', 'F1', 'MCC', 'Brier', 'Log-Loss', 'AUC-PR']

results_df = pd.DataFrame(index=models, columns=metrics)
# Preencher com resultados...
```

**Entregável**: Tabela comparativa completa

#### Bloco 2 (18:00-20:00): Gráficos
- [ ] Gráficos de acurácia ao longo do tempo
- [ ] Box plots de métricas
- [ ] Curvas de calibração

#### Bloco 3 (20:00-22:00): Análise de Erros
- [ ] Identificar períodos problemáticos
- [ ] Analisar casos de falha
- [ ] Correlacionar com eventos de mercado

**Entregável**: Análise de erros documentada

---

### **Sexta-feira, 07/02 (DIA 17)**
**Tema**: Backtesting - Parte 1

#### Bloco 1 (16:00-18:00): Implementar Backtester
- [ ] Completar `src/utils/backtesting.py`
- [ ] Incluir custos de transação
- [ ] Implementar slippage

```python
class SimpleBacktest:
    def __init__(self, costs={'corretagem': 10, 'taxa': 0.0003, 'slippage': 0.0001}):
        self.costs = costs
    
    def run(self, df, signals, capital_inicial=100000):
        position = 0
        cash = capital_inicial
        portfolio_value = [cash]
        trades = []
        
        for i, signal in enumerate(signals):
            price = df.iloc[i]['close']
            
            # Custos
            cost_fixo = self.costs['corretagem']
            cost_prop = price * self.costs['taxa']
            price_exec = price * (1 + self.costs['slippage'] * np.sign(signal))
            
            if signal == 1 and position == 0:  # Compra
                shares = (cash - cost_fixo) / price_exec
                position = shares
                cash -= (shares * price_exec + cost_fixo)
                trades.append({'type': 'BUY', 'price': price_exec, 'shares': shares})
                
            elif signal == -1 and position > 0:  # Venda
                cash += (position * price_exec - cost_fixo - cost_prop * position * price_exec)
                trades.append({'type': 'SELL', 'price': price_exec, 'shares': position})
                position = 0
            
            total = cash + (position * price if position > 0 else 0)
            portfolio_value.append(total)
        
        return self._calculate_metrics(portfolio_value, trades)
```

**Entregável**: Backtester funcionando

#### Bloco 2 (18:00-20:00): Backtest Baselines
- [ ] Rodar backtest para Naive
- [ ] Rodar backtest para ARIMA
- [ ] Calcular métricas de trading (Sharpe, Drawdown, etc.)

#### Bloco 3 (20:00-22:00): Backtest Modelos DL
- [ ] Rodar backtest para LSTM
- [ ] Rodar backtest para CNN-LSTM
- [ ] Comparar todos os modelos

**Entregável**: Backtests completos

---

### **Sábado, 08/02 (DIA 18)**
**Tema**: Análise de Backtests

#### Manhã (09:00-13:00): Métricas Financeiras
- [ ] Tabela: Retorno, Sharpe, Max DD, Turnover
- [ ] Curvas de equity
- [ ] Drawdown ao longo do tempo

```python
# Métricas de trading
def calculate_trading_metrics(portfolio_value, trades):
    returns = pd.Series(portfolio_value).pct_change().dropna()
    
    metrics = {
        'final_value': portfolio_value[-1],
        'return_pct': (portfolio_value[-1] / portfolio_value[0] - 1) * 100,
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252*26),
        'max_drawdown': (pd.Series(portfolio_value) / 
                         pd.Series(portfolio_value).cummax() - 1).min() * 100,
        'num_trades': len(trades),
        'turnover': calculate_turnover(trades)
    }
    
    return metrics
```

**Entregável**: Tabela de métricas financeiras

#### Tarde (14:00-18:00): Análise de Sensibilidade
- [ ] Testar diferentes thresholds de entrada
- [ ] Variar custos de transação
- [ ] Analisar impacto de slippage

#### Noite (20:00-22:00): Visualizações
- [ ] Gráficos comparativos
- [ ] Relatório de backtest
- [ ] Documentar insights

**Entregável**: Relatório de backtests completo

---

### **Domingo, 09/02 (DIA 19)**
**Tema**: Generalização e Testes Adicionais

#### Manhã (09:00-13:00): Outros Ativos
- [ ] Testar CNN-LSTM em VALE3
- [ ] Testar CNN-LSTM em ITUB4
- [ ] Comparar performance

#### Tarde (14:00-18:00): Análise Comparativa
- [ ] Performance por ativo
- [ ] Características que afetam resultado
- [ ] Documentar padrões

#### Noite (20:00-22:00): Consolidação
- [ ] Atualizar `PROGRESSO.md`
- [ ] Preparar material para Semana 4
- [ ] Revisar cronograma

**Entregável**: Análise multi-ativos completa

---

### **Segunda-feira, 10/02 (DIA 20)**
**Tema**: Preparação para Testes Estatísticos

#### Bloco 1 (16:00-18:00): Implementar Diebold-Mariano
- [ ] Criar função para teste DM
- [ ] Preparar dados de erro

```python
from scipy import stats

def diebold_mariano_test(errors_1, errors_2):
    """
    Teste de Diebold-Mariano para comparar acurácia preditiva.
    
    H0: Modelos têm mesma acurácia
    H1: Modelo 1 é diferente de Modelo 2
    """
    d = errors_1**2 - errors_2**2
    mean_d = d.mean()
    var_d = d.var()
    
    DM_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(DM_stat)))
    
    return {
        'statistic': DM_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

**Entregável**: Função DM implementada

#### Bloco 2 (18:00-20:00): Preparar Análise de Regimes
- [ ] Implementar detector de volatilidade
- [ ] Segmentar dados por regime
- [ ] Preparar pipeline de análise

```python
def detect_volatility_regime(returns, window=20, threshold=0.015):
    """
    Classifica períodos em regimes de volatilidade.
    
    - Alta volatilidade: vol > threshold
    - Baixa volatilidade: vol <= threshold
    """
    vol = returns.rolling(window=window).std()
    regime = (vol > threshold).astype(int)
    regime = regime.replace({0: 'low_vol', 1: 'high_vol'})
    return regime
```

**Entregável**: Pipeline de regimes pronto

#### Bloco 3 (20:00-22:00): Documentação
- [ ] Documentar semana 3
- [ ] Preparar checklist Semana 4
- [ ] Revisar objetivos

---

### **Terça-feira, 11/02 (DIA 21)**
**Tema**: Fechamento Semana 3

#### Bloco 1 (16:00-18:00): Consolidação
- [ ] Revisar todos os resultados
- [ ] Verificar reprodutibilidade
- [ ] Checar logs e outputs

#### Bloco 2 (18:00-20:00): Relatório Semanal
- [ ] Preparar slides para orientador
- [ ] Resumo executivo da semana
- [ ] Próximos passos

#### Bloco 3 (20:00-22:00): Planning
- [ ] Atualizar `PROGRESSO.md`
- [ ] Planejar Semana 4 em detalhes
- [ ] Revisar milestones

**Entregável**:
- ✅ CNN-LSTM otimizado e treinado
- ✅ Comparação completa com todos os baselines
- ✅ Backtests com custos de transação
- ✅ Análise multi-ativos
- ✅ Preparação para testes estatísticos

---

## 🗓️ SEMANA 4: TESTES ESTATÍSTICOS E ROBUSTEZ (12-18 Fev)

### 🎯 Objetivo da Semana
Realizar testes de significância estatística, análise de robustez por regimes de volatilidade, e análises de sensibilidade.

---

### **Quarta-feira, 12/02 (DIA 22)**
**Tema**: Testes de Diebold-Mariano

#### Bloco 1 (16:00-18:00): CNN-LSTM vs Baselines
- [ ] DM: CNN-LSTM vs Naive
- [ ] DM: CNN-LSTM vs ARIMA
- [ ] DM: CNN-LSTM vs Prophet
- [ ] Salvar resultados com p-values

```python
# Exemplo de uso
errors_cnn_lstm = y_true - y_pred_cnn_lstm
errors_arima = y_true - y_pred_arima

result = diebold_mariano_test(errors_cnn_lstm, errors_arima)
print(f"DM Statistic: {result['statistic']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Significativo? {result['significant']}")
```

**Entregável**: Tabela de testes DM

#### Bloco 2 (18:00-20:00): CNN-LSTM vs LSTM
- [ ] DM: CNN-LSTM vs LSTM puro
- [ ] Analisar se CNN adiciona valor
- [ ] Documentar insights

#### Bloco 3 (20:00-22:00): Consolidação
- [ ] Criar tabela consolidada de p-values
- [ ] Gerar gráfico de significância
- [ ] Interpretar resultados

**Entregável**: Relatório de testes estatísticos

---

### **Quinta-feira, 13/02 (DIA 23)**
**Tema**: Análise por Regimes de Volatilidade

#### Bloco 1 (16:00-18:00): Segmentação
- [ ] Detectar regimes de alta/baixa volatilidade
- [ ] Segmentar dados de teste
- [ ] Calcular métricas por regime

```python
# Análise por regime
regimes = detect_volatility_regime(returns, window=20, threshold=0.015)

for regime in ['low_vol', 'high_vol']:
    mask = regimes == regime
    y_true_regime = y_true[mask]
    y_pred_regime = y_pred[mask]
    
    metrics_regime = compute_all_metrics(y_true_regime, y_pred_regime)
    print(f"\n{regime}:")
    for metric, value in metrics_regime.items():
        print(f"  {metric}: {value:.4f}")
```

**Entregável**: Métricas por regime

#### Bloco 2 (18:00-20:00): Comparação
- [ ] Tabela: Modelo × Regime × Métrica
- [ ] Gráficos comparativos
- [ ] Identificar padrões

#### Bloco 3 (20:00-22:00): Análise de Crises
- [ ] Identificar períodos de crise (Mar/2020 - COVID)
- [ ] Performance durante choques
- [ ] Robustez a eventos extremos

**Entregável**: Análise de robustez completa

---

### **Sexta-feira, 14/02 (DIA 24)**
**Tema**: Análises de Sensibilidade

#### Bloco 1 (16:00-18:00): Sensibilidade a Janelas
- [ ] Testar com janelas de 5 minutos
- [ ] Testar com janelas de 30 minutos
- [ ] Comparar com baseline de 15 minutos

**Entregável**: Análise de granularidade temporal

#### Bloco 2 (18:00-20:00): Sensibilidade a Features
- [ ] Remover MME e retreinar
- [ ] Remover RSI e retreinar
- [ ] Remover Bollinger e retreinar
- [ ] Identificar features mais importantes

```python
# Ablation study
feature_sets = {
    'full': ['ema_9', 'ema_21', 'ema_50', 'rsi', 'bb_upper', 'bb_lower', 'volatility'],
    'no_ema': ['rsi', 'bb_upper', 'bb_lower', 'volatility'],
    'no_rsi': ['ema_9', 'ema_21', 'ema_50', 'bb_upper', 'bb_lower', 'volatility'],
    'no_bb': ['ema_9', 'ema_21', 'ema_50', 'rsi', 'volatility'],
}

for name, features in feature_sets.items():
    # Retreinar modelo...
    print(f"{name}: Accuracy = {accuracy:.4f}")
```

**Entregável**: Análise de importância de features

#### Bloco 3 (20:00-22:00): Sensibilidade a Custos
- [ ] Variar corretagem (5, 10, 20 R$)
- [ ] Variar slippage (0.01%, 0.05%, 0.1%)
- [ ] Analisar breakeven

**Entregável**: Análise de sensibilidade a custos

---

### **Sábado, 15/02 (DIA 25)**
**Tema**: Consolidação de Análises

#### Manhã (09:00-13:00): Consolidar Todos os Resultados
- [ ] Revisar todas as análises
- [ ] Criar tabelas consolidadas
- [ ] Gerar todos os gráficos necessários

**Entregável**: Pacote completo de resultados

#### Tarde (14:00-18:00): Interpretação
- [ ] Escrever interpretações
- [ ] Conectar com literatura
- [ ] Documentar limitações

#### Noite (20:00-22:00): Preparação para Escrita
- [ ] Organizar materiais
- [ ] Estruturar Capítulo de Resultados
- [ ] Listar tabelas e figuras necessárias

**Entregável**: Estrutura do Capítulo de Resultados

---

### **Domingo, 16/02 (DIA 26)**
**Tema**: Início da Escrita

#### Manhã (09:00-13:00): Seção 5.1 - Descrição dos Dados
- [ ] Escrever estatísticas descritivas
- [ ] Inserir tabelas
- [ ] Gráficos de séries temporais

**Entregável**: Seção 5.1 escrita

#### Tarde (14:00-18:00): Seção 5.2 - Desempenho Preditivo
- [ ] Escrever análise de métricas
- [ ] Inserir tabela comparativa
- [ ] Gráficos de evolução

**Entregável**: Seção 5.2 escrita

#### Noite (20:00-22:00): Continuar Escrita
- [ ] Revisar seções escritas
- [ ] Ajustar formatação ABNT
- [ ] Verificar citações

---

### **Segunda-feira, 17/02 (DIA 27)**
**Tema**: Continuar Escrita

#### Bloco 1 (16:00-18:00): Seção 5.3 - Desempenho Operacional
- [ ] Escrever análise de backtests
- [ ] Inserir tabelas de Sharpe, Drawdown
- [ ] Curvas de equity

**Entregável**: Seção 5.3 escrita

#### Bloco 2 (18:00-20:00): Seção 5.4 - Robustez
- [ ] Escrever análise de regimes
- [ ] Inserir resultados de DM
- [ ] Sensibilidades

**Entregável**: Seção 5.4 escrita

#### Bloco 3 (20:00-22:00): Seção 5.5 - Discussão
- [ ] Interpretar resultados
- [ ] Conectar com objetivos
- [ ] Discutir limitações

**Entregável**: Seção 5.5 escrita

---

### **Terça-feira, 18/02 (DIA 28)**
**Tema**: Fechamento Semana 4

#### Bloco 1 (16:00-18:00): Capítulo 6 - Conclusão
- [ ] Resumir contribuições
- [ ] Trabalhos futuros
- [ ] Considerações finais

**Entregável**: Conclusão escrita

#### Bloco 2 (18:00-20:00): Revisão Geral
- [ ] Revisar todos os capítulos
- [ ] Verificar coerência
- [ ] Ajustar transições

#### Bloco 3 (20:00-22:00): Formatação
- [ ] Aplicar normas ABNT
- [ ] Verificar referências
- [ ] Numerar tabelas e figuras

**Entregável**:
- ✅ Testes estatísticos completos
- ✅ Análise de robustez
- ✅ Análises de sensibilidade
- ✅ Capítulos de Resultados e Conclusão escritos

---

## 🗓️ SEMANA 5: FINALIZAÇÃO (19-20 Fev)

### 🎯 Objetivo da Semana
Finalizar escrita, revisar monografia completa, preparar slides de defesa, fazer ensaio.

---

### **Quarta-feira, 19/02 (DIA 29)**
**Tema**: Revisão Final

#### Bloco 1 (16:00-18:00): Revisão ABNT
- [ ] Verificar formatação completa
- [ ] Conferir margens, fontes, espaçamentos
- [ ] Revisar sumário
- [ ] Verificar paginação

```markdown
# Checklist ABNT
- [ ] Capa
- [ ] Folha de rosto
- [ ] Resumo (PT) - máx 500 palavras
- [ ] Abstract (EN) - máx 500 palavras
- [ ] Sumário automático
- [ ] Listas de figuras, tabelas, abreviaturas
- [ ] Corpo do texto: fonte 12, Times New Roman
- [ ] Legendas: fonte 10
- [ ] Margens: 3cm (esq/sup), 2cm (dir/inf)
- [ ] Espaçamento: 1.5 linhas
- [ ] Referências: ABNT NBR 6023
- [ ] Citações: ABNT NBR 10520
```

**Entregável**: Documento formatado ABNT

#### Bloco 2 (18:00-20:00): Revisão de Conteúdo
- [ ] Revisar introdução e objetivos
- [ ] Verificar se todos os objetivos foram atingidos
- [ ] Checar coerência entre seções
- [ ] Revisar ortografia e gramática

#### Bloco 3 (20:00-22:00): Preparar Slides
- [ ] Criar estrutura da apresentação (15-20 slides)
- [ ] Selecionar gráficos e tabelas principais
- [ ] Escrever roteiro

```markdown
# Estrutura Slides (15-20 min)
1. Introdução (2 slides)
   - Contexto e motivação
   - Problema de pesquisa
   
2. Objetivos (1 slide)
   - Objetivo geral e específicos
   
3. Fundamentação (3 slides)
   - Mercado financeiro e B3
   - LSTM e CNN
   - Walk-forward validation
   
4. Metodologia (3 slides)
   - Dados (ativos, período, features)
   - Arquiteturas testadas
   - Protocolo experimental
   
5. Resultados (6 slides)
   - Tabela comparativa principal
   - Gráfico de acurácia ao longo tempo
   - Backtests (Sharpe, Drawdown)
   - Testes estatísticos (DM)
   - Análise de robustez
   - Síntese dos resultados
   
6. Discussão (2 slides)
   - Por que CNN-LSTM superou (ou não)?
   - Limitações do estudo
   
7. Conclusão (2 slides)
   - Contribuições
   - Trabalhos futuros
```

**Entregável**: Slides prontos

---

### **Quinta-feira, 20/02 (DIA 30)**
**Tema**: Entrega e Preparação para Defesa

#### Bloco 1 (16:00-18:00): Revisão Final da Monografia
- [ ] Leitura completa end-to-end
- [ ] Corrigir últimos erros
- [ ] Gerar PDF final
- [ ] Verificar hyperlinks (se aplicável)

**Entregável**: PDF final pronto

#### Bloco 2 (18:00-20:00): Ensaio de Apresentação
- [ ] Ensaiar apresentação cronometrada (10-15 min)
- [ ] Ajustar timing dos slides
- [ ] Preparar respostas para perguntas esperadas

```markdown
# Perguntas Esperadas
1. Por que CNN-LSTM e não Transformers?
2. Como você garante que não há data leakage?
3. Os custos de transação estão realistas?
4. Por que a acurácia não foi maior?
5. Qual a aplicabilidade prática?
6. Trabalhos futuros mais específicos?
```

**Entregável**: Ensaio completo

#### Bloco 3 (20:00-22:00): Upload e Preparativos
- [ ] Fazer upload da monografia
- [ ] Enviar para orientador
- [ ] Preparar materiais auxiliares (anexos, códigos)
- [ ] Fazer backup de tudo

#### Final (22:00-23:00): Celebração e Descanso
- [ ] Revisar jornada dos 30 dias
- [ ] Comemorar conquista! 🎉
- [ ] Descansar bem antes da defesa

**Entregável**:
- ✅ Monografia ENTREGUE
- ✅ Slides prontos
- ✅ Ensaio feito
- ✅ TCC2 COMPLETO! 🎓

---

## 📈 Indicadores de Sucesso

### Métricas de Progresso (Atualizar Diariamente)

```markdown
# PROGRESSO.md

## Semana 1: [█████████░] 90%
- Dados: ✅
- Features: ✅
- Baselines: ✅
- Walk-forward: ✅
- Ambiente DL: 🔄

## Semana 2: [░░░░░░░░░░] 0%
- LSTM implementado: ⏳
- Optuna: ⏳
- Resultados: ⏳

## Semana 3: [░░░░░░░░░░] 0%
...

## Semana 4: [░░░░░░░░░░] 0%
...

## Semana 5: [░░░░░░░░░░] 0%
...
```

### KPIs do Projeto

| KPI | Meta | Atual | Status |
|-----|------|-------|--------|
| Dias trabalhados | 30 | 0 | 🟡 |
| Horas efetivas | 180h | 0h | 🟡 |
| Modelos implementados | 5 | 0 | 🟡 |
| Folds walk-forward | >10 | 0 | 🟡 |
| Ativos testados | 3 | 0 | 🟡 |
| Páginas escritas | 40+ | 0 | 🟡 |
| Tabelas/Figuras | 15+ | 0 | 🟡 |
| Acurácia CNN-LSTM | >55% | - | ⏳ |
| p-value DM | <0.05 | - | ⏳ |

---

## ⚠️ Plano de Contingência

### Se Atrasar >2 dias
1. **Reduzir escopo**:
   - Focar em 1 ativo principal (PETR4)
   - Reduzir trials Optuna: 50 → 30
   - Simplificar análises de sensibilidade

2. **Priorizar**:
   - CNN-LSTM > LSTM > Baselines
   - Métricas principais > Métricas secundárias
   - Escrita > Slides perfeitos

3. **Pedir ajuda**:
   - Comunicar orientador IMEDIATAMENTE
   - Solicitar extensão de prazo (se possível)
   - Negociar reduções de escopo

### Se Modelos Não Convergirem
1. Simplificar arquitetura
2. Usar hiperparâmetros da literatura
3. Testar em dados sintéticos primeiro
4. Procurar debugging em fóruns (Stack Overflow, Reddit)

### Se GPU Falhar
1. Usar Google Colab Pro ($10/mês)
2. AWS EC2 spot instances
3. Reduzir batch size drasticamente
4. Usar CPU (último recurso, muito lento)

---

## 🎯 Mantra Final

> **"Um dia de cada vez. Um bloco de cada vez. Uma linha de código de cada vez."**
>
> **"Progresso > Perfeição. Entregue > Esperando. Feito > Pensando."**
>
> **"30 dias. 180 horas. 1 objetivo: TERMINAR O TCC!"**

---

**🔥 COMEÇAR EM: 22/01/2026 às 16:00**  
**🏁 ENTREGAR EM: 20/02/2026 às 20:00**  
**⏱️ TEMPO RESTANTE: 30 DIAS**

**VAMOS FAZER HISTÓRIA! 💪🚀📚**
