# Arquitetura dos Modelos de Deep Learning

**Data:** 2025-01-23  
**Status:** Referência (estrutura `src/models/`)

---

## 1. Estrutura do Projeto

### Organização dos Modelos

```
src/models/
├── baselines.py          # Modelos baseline (Naive, Drift, ARIMA)
├── prophet_model.py      # Baseline Prophet
├── lstm_model.py         # Modelo LSTM puro (Baseline 3)
└── cnn_lstm_model.py    # Modelo híbrido CNN-LSTM (Modelo Principal)
```

### Por que essa organização?

1. **Separação de responsabilidades**: Cada modelo em seu próprio arquivo
2. **Facilita manutenção**: Fácil encontrar e modificar um modelo específico
3. **Reutilização**: Modelos podem ser importados em diferentes scripts
4. **Clareza**: Fica explícito quais modelos existem no projeto

---

## 2. Modelos Implementados

### 1. Baselines (Estatísticos/Simples)

#### `baselines.py`
- **NaiveBaseline**: Repete a última direção observada
- **DriftBaseline**: Assume tendência linear (drift)
- **ARIMABaseline**: Modelo estatístico Box-Jenkins (ARIMA)

#### `prophet_model.py`
- **ProphetBaseline**: Modelo de decomposição aditiva com sazonalidades (Facebook Prophet)

**Uso**: Comparação e baseline para avaliar se modelos complexos valem a pena.

---

### 2. Modelos de Deep Learning

#### `lstm_model.py` - Modelo LSTM Puro (Baseline 3)

**Arquitetura**:
```
Input: (n_steps=60, n_features=12)
  ↓
LSTM(50 unidades, dropout=0.2)
  ↓
Dense(1, sigmoid) → Classificação binária
```

**Função**: `criar_modelo_lstm()`

**Por que existe?**
- Baseline para comparar com o modelo híbrido CNN-LSTM
- Testa se apenas LSTM já é suficiente
- Permite avaliar o ganho da adição de CNN

**Quando usar**: Para comparação com CNN-LSTM e como baseline de deep learning.

---

#### `cnn_lstm_model.py` - Modelo Híbrido CNN-LSTM (Modelo Principal)

**Arquitetura**:
```
Input: (n_steps=60, n_features=12)
  ↓
Conv1D(64 filtros, kernel_size=2) → Extrai padrões locais
  ↓
MaxPooling1D(pool_size=2) → Reduz dimensionalidade
  ↓
LSTM(50 unidades, dropout=0.2) → Captura dependências temporais
  ↓
Dense(1, sigmoid) → Classificação binária
```

**Função**: `criar_modelo_cnn_lstm()`

**Por que essa arquitetura?**
1. **CNN (Conv1D)**: Captura padrões locais em janelas curtas (ex: reversão em 2-3 barras)
2. **MaxPooling**: Reduz dimensionalidade e aumenta robustez a ruído
3. **LSTM**: Captura dependências temporais de longo prazo (ex: tendências de 10-20 barras)
4. **Dense**: Classificação binária final (alta/baixa)

**Justificativa Técnica**:
- Dados financeiros têm padrões em múltiplas escalas temporais
- CNN é eficiente para padrões locais (curto prazo)
- LSTM é eficiente para dependências temporais (longo prazo)
- A combinação permite capturar ambos os tipos de padrão

**Quando usar**: Modelo principal do TCC, usado para predição final.

---

## 3. Como os Modelos são Usados

### Script de Treinamento: `train.py`

O script `train.py` importa os modelos e os treina:

```python
from src.models.lstm_model import criar_modelo_lstm
from src.models.cnn_lstm_model import criar_modelo_cnn_lstm

# Criar modelo
model = criar_modelo_cnn_lstm(n_steps=60, n_features=12)

# Treinar com walk-forward validation
# ... (código de treinamento)
```

### Fluxo Completo

1. **Carregar dados** → `load_data.py`
2. **Criar features** → `feature_engineering.py`
3. **Preparar sequências** → `prepare_sequences.py`
4. **Criar modelo** → `lstm_model.py` ou `cnn_lstm_model.py`
5. **Treinar** → `train.py` (com walk-forward validation)
6. **Avaliar** → Métricas em `metrics.py`

---

## 4. Parâmetros dos Modelos

### LSTM
- `n_steps`: 60 barras (janela temporal)
- `n_features`: 12 features (EMA, RSI, Bollinger, Volatilidade)
- `lstm_units`: 50 unidades
- `dropout`: 0.2 (regularização)
- `learning_rate`: 0.001

### CNN-LSTM
- `n_steps`: 60 barras
- `n_features`: 12 features
- `conv_filters`: 64 filtros
- `conv_kernel_size`: 2 (padrões em 2 barras)
- `pool_size`: 2 (redução pela metade)
- `lstm_units`: 50 unidades
- `dropout`: 0.2
- `learning_rate`: 0.001

---

## 5. Como Explicar para o Professor

### Estrutura do Projeto
"Organizei os modelos em arquivos separados para facilitar manutenção e reutilização. Cada modelo tem sua própria implementação seguindo o padrão do projeto."

### Modelos Baseline
"Implementei 5 baselines para comparação:
1. Naive, Drift, ARIMA (estatísticos)
2. Prophet (decomposição aditiva)
3. LSTM puro (baseline de deep learning)"

### Modelo Principal
"O modelo principal é o CNN-LSTM híbrido, que combina:
- CNN para padrões locais (curto prazo)
- LSTM para dependências temporais (longo prazo)

Isso permite capturar padrões em múltiplas escalas temporais, o que é importante para dados financeiros intradiários."

### Justificativa Técnica
"A escolha da arquitetura CNN-LSTM é baseada em:
1. Literatura sobre séries temporais financeiras
2. Capacidade de capturar padrões em múltiplas escalas
3. Comparação com LSTM puro para validar o ganho da híbrida"

---

## 6. Referências no TCC

- **Seção 4.3**: Arquitetura dos Modelos
- **Seção 3.2**: Baselines para Comparação
- **Seção 4.4**: Desenho Experimental e Treinamento
