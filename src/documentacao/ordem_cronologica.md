# Ordem Cronológica - Desenvolvimento do TCC

Documentação cronológica de todas as decisões técnicas, implementações e análises realizadas.

---

## 2026-01-23 - Correções Críticas no Treinamento

### Contexto
- Primeiro treinamento do modelo CNN-LSTM com Optuna
- Identificados problemas que impediam aprendizado adequado
- Acurácias muito baixas (~50-54%) indicando problemas

### Problemas Identificados

1. **BUG CRÍTICO**: Banda morta não aplicada
   - Função chamada sem parâmetro `threshold`
   - Usando valor padrão 0.0 ao invés de 0.001
   - Resultado: apenas 4.6% neutros (deveria ser ~40%)

2. **Threshold inadequado**
   - Threshold de 0.05% muito pequeno para dados intradiários
   - Classificando ruído como movimento significativo

3. **Convergência insuficiente**
   - Patience muito baixo (5 épocas)
   - Poucas épocas máximas (30)
   - Modelos não convergiam adequadamente

### Correções Aplicadas

1. **Aplicação correta da banda morta**
   - Adicionado `threshold=THRESHOLD_BANDA_MORTA` na chamada
   - Threshold aumentado de 0.0005 para 0.001 (0.1%)

2. **Ajustes de hiperparâmetros**
   - Patience aumentado: 5 → 10 épocas
   - Épocas máximas: 30 → 100
   - ReduceLROnPlateau patience: 3 → 5

### Resultados

**Melhorias:**
- ✅ Neutros: 4.6% → **42.8%** (correto!)
- ✅ Maior variância nas probabilidades (std: 0.006 → 0.010)
- ✅ Acurácia melhorou: ~50% → ~53%

**Problemas ainda existentes:**
- ⚠️ Acurácia ainda baixa (~53%) - possível limitação do mercado
- 🔴 Alguns modelos colapsando para "sempre prever baixa"
- ⚠️ Learning rates altos (0.01) causando convergência prematura

### Interpretação

- Acurácia de 53% é considerada **boa** na literatura de finanças quantitativas
- Acima de 50% indica poder preditivo real
- Movimentos intradiários são notoriamente difíceis de prever

### Arquivos Modificados
- `src/config.py` - Aumentado THRESHOLD_BANDA_MORTA
- `src/data_processing/feature_engineering.py` - Aplicado threshold corretamente
- `src/utils/optuna_optimizer.py` - Ajustes de convergência
- `src/train.py` - Aumentado épocas padrão

### Documentação
- [Correções do Treinamento](implementacoes/correcoes_treinamento_2026_01_23.md) - Documentação completa

### Próximos Passos
- Avaliar resultados completos dos 5 folds
- Ajustar espaço de busca do Optuna (remover lr=0.01)
- Testar outras arquiteturas se necessário

---

## 2025-01-23 - Remoção da Banda Morta

### Contexto
- Banda morta original: ±0.0005 (0.05%)
- 22.3% dos dados classificados como neutros
- Apenas 4.6% são realmente zero
- 6,225 amostras (17.7%) sendo perdidas

### Análise Realizada
- Total de retornos: 35,153
- Média: 0.000012
- Desvio-padrão: 0.003443
- Retornos dentro da banda morta: 7,848 (22.3%)
- Retornos realmente zero: 1,624 (4.6%)

### Decisão Tomada
- Remover banda morta (threshold = 0.0)
- Usar apenas sinal do retorno (>0, <0, ==0)
- Aplicar em: target creation, métricas, baselines

### Justificativa
- Perda de 17.7% dos dados era significativa
- Retornos intradiários são naturalmente pequenos
- Banda morta eliminava informações úteis para previsão
- Para previsão de direção, qualquer movimento é relevante

### Impacto
- +17.7% de amostras utilizadas (6,225 amostras recuperadas)
- ARIMA F1_Score melhorou: 0.576 → 0.593
- Métricas mais realistas usando quase todos os dados
- Distribuição de targets: Alta 38.2%, Baixa 39.5%, Neutro 4.6% (antes: 22.3% neutros)

### Arquivos Modificados
- `src/data_processing/feature_engineering.py` - criar_target_com_banda_morta()
- `src/utils/metrics.py` - calcular_acuracia_direcional(), calcular_metricas_preditivas()
- `src/models/baselines.py` - NaiveBaseline, DriftBaseline, ARIMABaseline

---

## 2025-01-23 - Correção do Problema ARIMA

### Contexto
- ARIMA retornando F1_Score = 0.0 e MCC = 0.0
- 100% das previsões eram zeros (neutros)

### Análise Realizada
- Forecasts do ARIMA muito pequenos: min=-0.000023, max=0.000004
- Todos os forecasts dentro da banda morta original (±0.0005)
- Threshold muito grande para valores tão pequenos

### Decisão Tomada
- Remover banda morta resolveu o problema
- ARIMA agora usa apenas sinal do forecast

### Justificativa
- Forecasts de retornos são naturalmente muito pequenos
- Banda morta impedia captura da direção
- Sinal do forecast é suficiente para classificação

### Impacto
- ARIMA passou a prever direções reais
- Distribuição: 1=1637, -1=546, 0=205 (antes: 0=2388)
- F1_Score: 0.0 → 0.593

---

## 2025-01-23 - Implementação Walk-Forward Validation

### Contexto
- Necessidade de validação temporal rigorosa
- Evitar data leakage em séries temporais financeiras

### Implementação
- Classe WalkForwardValidator criada
- Suporte a embargo temporal
- Divisão sequencial de dados
- Agregação de resultados por fold

### Características
- Treino: 6552 barras (~1 ano)
- Teste: 546 barras (~1 mês)
- Embargo: 1 barra
- Geração automática de folds

### Justificativa
- Validação walk-forward é obrigatória para séries temporais
- K-fold tradicional viola ordem temporal
- Embargo previne contaminação entre treino/teste

### Arquivos Criados
- `src/utils/validation.py` - WalkForwardValidator, FoldInfo

---

## 2025-01-23 - Simplificação do Código

### Contexto
- Código muito modularizado e verboso
- Muitos fallbacks desnecessários
- Comentários excessivos

### Decisão Tomada
- Remover todos os fallbacks de import
- Simplificar docstrings
- Reduzir comentários excessivos
- Manter apenas código essencial

### Impacto
- Redução de ~50% nas linhas de código
- Código mais legível e direto
- Imports consistentes (apenas relativos)
- Manutenção mais fácil

### Arquivos Simplificados
- `testar_baselines_walkforward.py`: 277 → 141 linhas
- `load_data.py`: 325 → 120 linhas
- `feature_engineering.py`: 449 → 124 linhas
- `baselines.py`: 328 → 135 linhas
- `metrics.py`: 192 → 66 linhas
- `validation.py`: 413 → 180 linhas

---

## 2025-01-23 - Implementação de Baselines

### Implementação
- NaiveBaseline: repete última direção
- DriftBaseline: projeta tendência linear
- ARIMABaseline: modelo Box-Jenkins com grid search

### Características
- Interface comum (BaseBaseline)
- Otimização ARIMA por AIC
- Conversão de forecasts para direções

### Resultados Iniciais (com banda morta)
- Naive: 50.95% acurácia
- Drift: 49.37% acurácia
- ARIMA: 50.95% acurácia (mas F1=0.0)

### Resultados Finais (sem banda morta)
- Naive: 50.50% acurácia, F1=0.315
- Drift: 49.76% acurácia, F1=0.543
- ARIMA: 48.36% acurácia, F1=0.593

### Arquivos Criados
- `src/models/baselines.py`

---

## 2025-01-23 - Engenharia de Features

### Features Implementadas
- Retornos logarítmicos
- EMAs: 9, 21, 50 períodos
- RSIs: 9, 21, 50 períodos
- Bandas de Bollinger (20 períodos, 2 desvios)
- Volatilidade realizada (20 períodos)
- Target de direção

### Justificativa
- Features técnicas padrão em análise financeira
- Múltiplos períodos para capturar diferentes escalas temporais
- Target binário (alta/baixa) para classificação

### Arquivos Criados
- `src/data_processing/feature_engineering.py`

---

## 2025-01-23 - Implementação de Métricas

### Métricas Implementadas
- Acurácia direcional
- Acurácia, Balanced Accuracy
- F1-Score, MCC
- Brier Score, Log-Loss, AUC-PR (quando disponível)

### Características
- Sem banda morta (ignora apenas zeros reais)
- Foco em métricas robustas a desbalanceamento
- Suporte a métricas probabilísticas

### Arquivos Criados
- `src/utils/metrics.py`

---

## 2025-01-23 - Estrutura de Configuração

### Decisão
- Centralizar todas as configurações em `src/config.py`
- Remover fallbacks de import
- Usar apenas imports relativos

### Configurações Centralizadas
- Estrutura de dados (colunas obrigatórias)
- Horário de pregão B3
- Períodos de indicadores técnicos
- Tamanhos de walk-forward
- Custos de transação
- Seed para reprodutibilidade

### Arquivos Criados
- `src/config.py`

---

## 2025-01-26 - Resultados dos Baselines com Walk-Forward

### Contexto
- Implementação completa de 4 baselines: Naive, Drift, ARIMA, Prophet
- Teste com walk-forward validation em VALE3
- 5 folds, 2,388 amostras de teste agregadas

### Resultados Obtidos

| Baseline | Accuracy Direcional | F1-Score | MCC |
|----------|---------------------|----------|-----|
| Naive | 50.50% | 0.315 | 0.002 |
| Drift | 49.76% | 0.543 | -0.002 |
| ARIMA | 48.36% | 0.593 | -0.029 |
| Prophet | 50.50% | 0.531 | 0.012 |

### Análise e Interpretação

**Performance Geral:**
- Todos os baselines performam próximo de 50% (aleatório)
- Isso é **esperado e desejável** para baselines simples
- Confirma que predizer direção de preços é um problema difícil

**Destaques:**
- Naive e Prophet: melhor acurácia direcional (50.50%)
- ARIMA: melhor F1-Score (0.593)
- Prophet: melhor MCC (0.012) - correlação positiva, ainda que fraca

**Validação Metodológica:**
- ✅ Walk-forward funcionou corretamente (sem data leakage)
- ✅ Baseline estabelecido (~50%) para comparação com deep learning
- ✅ Resultados documentados e prontos para TCC

### Justificativa para Deep Learning
- Baselines simples não superam o acaso
- Modelos não-lineares (LSTM, CNN-LSTM) podem capturar padrões complexos
- Expectativa: modelos de deep learning devem superar 52-55% para serem úteis

### Arquivos Atualizados
- `src/documentacao/implementacoes/baselines.md` - Documentação completa dos resultados
- `data/processed/VALE3_baselines_walkforward.csv` - Resultados salvos

---
