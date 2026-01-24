# Correções Aplicadas ao Treinamento - 23/01/2026

## Problemas Identificados

### 1. Banda Morta Não Aplicada (BUG CRÍTICO)
**Problema**: A função `criar_target_com_banda_morta()` estava sendo chamada sem o parâmetro `threshold`, usando o valor padrão 0.0 ao invés de `THRESHOLD_BANDA_MORTA = 0.0005`.

**Impacto**: Apenas retornos EXATAMENTE zero eram classificados como neutros, resultando em apenas 4.6% de neutros. Isso incluía muito ruído nos dados de treinamento.

**Correção**: Adicionado `threshold=THRESHOLD_BANDA_MORTA` na chamada da função em `feature_engineering.py` linha 106.

### 2. Banda Morta Muito Pequena
**Problema**: Threshold de 0.05% era muito pequeno para movimentos de 15 minutos, classificando ruído como movimentos significativos.

**Impacto**: Modelo tentava prever movimentos aleatórios ao invés de tendências reais, resultando em acurácias próximas de 50% (chute aleatório).

**Correção**: Aumentado threshold de 0.0005 (0.05%) para 0.001 (0.1%) em `config.py`.

### 3. Convergência Insuficiente
**Problema**: 
- Patience muito baixo (5 épocas) no early stopping
- Apenas 30 épocas máximas para treinar
- Poucos trials do Optuna (20)

**Impacto**: Modelos não tinham tempo suficiente para convergir, especialmente com learning rates baixas.

**Correção**:
- Aumentado patience de 5 para 10 épocas
- Aumentado épocas máximas de 30 para 100 (com early stopping)
- Aumentado padrão de trials do Optuna de 20 para 30

## Mudanças Realizadas

### `config.py`
```python
# ANTES
THRESHOLD_BANDA_MORTA = 0.0005  # 0.05%

# DEPOIS
THRESHOLD_BANDA_MORTA = 0.001  # 0.1%
```

### `feature_engineering.py`
```python
# ANTES
df_features['target'] = criar_target_com_banda_morta(df_features)

# DEPOIS
df_features['target'] = criar_target_com_banda_morta(df_features, threshold=THRESHOLD_BANDA_MORTA)

# Também atualizada a docstring da função para deixar claro o uso da banda morta
```

### `optuna_optimizer.py`
```python
# ANTES
epochs: int = 30
patience=5
patience=3

# DEPOIS
epochs: int = 100
patience=10
patience=5
```

### `train.py`
```python
# ANTES
parser.add_argument('--epochs', type=int, default=50)

# DEPOIS
parser.add_argument('--epochs', type=int, default=100)
```

## Resultados Esperados

Com essas correções, esperamos:

1. **Mais neutros removidos**: ~15-25% dos dados classificados como neutros ao invés de 4.6%
2. **Melhor aprendizado**: Modelo focará em movimentos significativos (> 0.1%)
3. **Maior convergência**: Mais épocas e patience adequado permitirão convergência completa
4. **Acurácias mais altas**: Esperamos acurácias de 55-60% ao invés de ~50-54%

## Como Testar

```bash
# Interromper treinamento atual (Ctrl+C)

# Executar novamente com as correções
uv run python src/train.py --ativo VALE3 --modelo cnn_lstm --optuna --n-trials 30

# Observar:
# 1. Percentual de neutros deve aumentar (~15-25%)
# 2. Acurácias devem melhorar (>55%)
# 3. Probabilidades devem ter maior variância (std > 0.01)
```

## Fundamentação Teórica

### Banda Morta
Conforme metodologia do TCC (Seção 4.2), a banda morta é essencial para:
- Filtrar ruído intradiário
- Focar em movimentos com significância estatística
- Evitar overfitting em flutuações aleatórias

**Referência**: Prado, M. L. (2018). "Advances in Financial Machine Learning" - Capítulo 3: Labeling

### Threshold Adequado
Para barras de 15 minutos:
- 0.05% é menor que spread típico (0.1-0.2%)
- 0.1% representa movimento mínimo significativo
- Estudos empíricos sugerem 0.1-0.3% para intradiário

**Referências**: 
- Lopez de Prado (2018)
- Bergmeir & Benítez (2012)
