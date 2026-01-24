# Implementação: Modelos Baseline

**Data:** 2025-01-23  
**Status:** Implementado e testado

---

## Baselines Implementados

### 1. NaiveBaseline
- **Estratégia:** Repete última direção observada
- **Treino:** Armazena direção do último retorno
- **Previsão:** Sempre retorna mesma direção
- **Uso:** Linha de base mínima (baseline 0)

### 2. DriftBaseline
- **Estratégia:** Projeta tendência linear (drift)
- **Treino:** Calcula média dos retornos históricos
- **Previsão:** Projeta usando drift acumulado
- **Uso:** Captura tendência de longo prazo

### 3. ARIMABaseline
- **Estratégia:** Modelo estatístico Box-Jenkins
- **Treino:** Grid search para otimizar (p,d,q) por AIC
- **Previsão:** Forecast de retornos, convertido para direção
- **Uso:** Baseline estatístico clássico

### 4. ProphetBaseline
- **Estratégia:** Decomposição aditiva com sazonalidades (Facebook Prophet)
- **Treino:** Ajusta componentes de tendência e sazonalidade
- **Previsão:** Forecast com sazonalidade diária e semanal
- **Uso:** Baseline para séries temporais com padrões sazonais

---

## Arquitetura

### Interface Comum
- Classe base: `BaseBaseline` (ABC)
- Métodos: `fit()`, `predict()`
- Todos retornam direções: 1 (alta), -1 (baixa), 0 (neutro)

### Características
- Sem banda morta (threshold = 0.0)
- Usam apenas sinal dos retornos/forecasts
- Compatíveis com walk-forward validation

---

## Resultados (Walk-Forward, VALE3)

**Data do Teste:** 2025-01-26  
**Ativo:** VALE3  
**Período:** 2020-2025 (barras de 15 minutos)

### Configuração
- **Folds:** 5
- **Amostras de teste:** 2,388 (total agregado)
- **Treino por fold:** 6,552 barras (~1 ano)
- **Teste por fold:** 546 barras (~1 mês)
- **Embargo:** 1 barra entre treino e teste

### Resultados Completos

| Baseline | N_Folds | N_Teste | Accuracy | Balanced Acc | F1_Score | MCC | Accuracy Direcional |
|----------|---------|---------|----------|--------------|----------|-----|---------------------|
| **Naive** | 5 | 2,388 | 50.50% | 50.09% | 0.315 | 0.002 | **50.50%** |
| **Drift** | 5 | 2,388 | 49.76% | 49.93% | 0.543 | -0.002 | 49.76% |
| **ARIMA** | 5 | 2,388 | 48.36% | 48.79% | **0.593** | -0.029 | 48.36% |
| **Prophet** | 5 | 2,388 | 50.50% | 50.60% | 0.531 | **0.012** | **50.50%** |

### Melhores por Métrica
- **Accuracy Direcional:** Naive e Prophet (50.50%)
- **Accuracy:** Naive e Prophet (50.50%)
- **F1-Score:** ARIMA (0.593)
- **MCC:** Prophet (0.012)

### Análise dos Resultados

#### 1. Performance Geral
- **Todos os baselines performam próximo de 50%** (acurácia direcional entre 48.36% e 50.50%)
- Isso é **esperado e desejável** para baselines simples em predição de direção de preços
- Confirma que predizer direção de preços é um problema difícil (mercado eficiente)

#### 2. Interpretação por Baseline

**Naive (50.50%)**
- Melhor acurácia direcional (empate com Prophet)
- Estratégia mais simples possível (repetir última direção)
- Serve como linha de base mínima

**Drift (49.76%)**
- Ligeiramente abaixo de 50%
- Captura tendência linear, mas pode ser enganado por reversões
- F1-Score razoável (0.543)

**ARIMA (48.36%)**
- Pior acurácia direcional, mas **melhor F1-Score (0.593)**
- Modelo estatístico clássico, mas pode ter dificuldade com não-linearidades
- MCC negativo indica correlação fraca com direção real

**Prophet (50.50%)**
- Melhor MCC (0.012) - correlação positiva, ainda que fraca
- Sazonalidade diária/semanal pode ajudar em padrões intradiários
- Empate em acurácia com Naive

#### 3. Validação Metodológica

✅ **Walk-forward funcionou corretamente**
- 5 folds gerados e executados
- Sem data leakage aparente
- Métricas calculadas consistentemente

✅ **Baseline estabelecido**
- ~50% é a referência para comparar modelos de deep learning
- Se LSTM/CNN-LSTM superarem 52-55%, será um ganho significativo

✅ **Resultados documentados**
- Útil para o TCC mostrar que baselines simples não superam o acaso
- Justifica a necessidade de modelos mais sofisticados

#### 4. Expectativas para Deep Learning

Com baselines em ~50%, esperamos que:
- **LSTM puro:** 52-55% (ganho modesto)
- **CNN-LSTM híbrido:** 53-58% (ganho mais significativo se capturar padrões não-lineares)

Qualquer resultado acima de 55% será considerado **excelente** para predição de direção intradiária.

---

## Problemas Encontrados e Resolvidos

### Problema: ARIMA prevendo 100% zeros
- **Causa:** Forecasts muito pequenos dentro da banda morta
- **Solução:** Remoção da banda morta
- **Resultado:** ARIMA passou a prever direções reais

---

## Referências para TCC

### Seção: Metodologia - Modelos Baseline

**Pontos a mencionar:**
- Quatro baselines implementados (Naive, Drift, ARIMA, Prophet)
- Interface comum (`BaseBaseline`) para facilitar comparação
- Otimização ARIMA por AIC (grid search)
- Prophet configurado com sazonalidade diária e semanal
- Conversão de forecasts para direções (1, -1, 0)

### Seção: Resultados - Comparação de Baselines

**Pontos a mencionar:**
- **Resultados de walk-forward validation** (5 folds, 2,388 amostras de teste)
- **Métricas calculadas:** acurácia, balanced accuracy, F1-Score, MCC, acurácia direcional
- **Interpretação principal:** Todos os baselines performam próximo de 50% (aleatório)
  - Isso é **esperado** para baselines simples em predição de direção de preços
  - Confirma a dificuldade do problema (mercado eficiente)
  - Estabelece linha de base para comparar modelos de deep learning
- **Destaques:**
  - Naive e Prophet: melhor acurácia direcional (50.50%)
  - ARIMA: melhor F1-Score (0.593)
  - Prophet: melhor MCC (0.012) - correlação positiva, ainda que fraca
- **Justificativa para Deep Learning:**
  - Baselines simples não superam o acaso
  - Modelos não-lineares (LSTM, CNN-LSTM) podem capturar padrões complexos
  - Expectativa: modelos de deep learning devem superar 52-55% para serem considerados úteis

---

## Arquivos

- `src/models/baselines.py` - Implementação (Naive, Drift, ARIMA)
- `src/models/prophet_model.py` - Implementação Prophet
- `src/tests/testar_baselines_walkforward.py` - Script de testes
- `data/processed/VALE3_baselines_walkforward.csv` - Resultados salvos

## Conclusão

Os resultados dos baselines validam:
1. ✅ **Metodologia correta:** Walk-forward funcionando sem data leakage
2. ✅ **Baseline estabelecido:** ~50% como referência para comparação
3. ✅ **Justificativa técnica:** Necessidade de modelos mais sofisticados (deep learning)
4. ✅ **Documentação completa:** Resultados prontos para inclusão no TCC

**Próximo passo:** Treinar e avaliar modelos LSTM e CNN-LSTM para comparar com esses baselines.
