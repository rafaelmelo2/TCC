# Ordem Cronológica - Desenvolvimento do TCC

**Data:** 2025-01-23 a 2026-02-16  
**Status:** Mantido (timeline de decisões e implementações)

Documentação cronológica de todas as decisões técnicas, implementações e análises realizadas.

---

## 2026-02-16 - Capítulos 5 e 6 do TCC com Dados Reais

### Contexto
- Capítulo 5 (Resultados e Discussão) e Capítulo 6 (Considerações Finais) precisavam ser preenchidos com dados reais do pipeline
- Dados consolidados em CSVs: comparativo, análise CNN-LSTM, DM, backtests
- Capítulo mais importante do PFC2

### O que foi feito
- Escrita completa do Capítulo 5 (~632 linhas) com todos os dados reais dos CSVs
- Reescrita completa do Capítulo 6 para refletir resultados reais
- Heatmap DM gerado/atualizado via `gerar_tabelas_graficos_dm.py`

### Decisões técnicas
1. **Estrutura Cap. 5**: 9 seções (Visão Geral, Comparação Geral, Estabilidade por Fold, Calibração de Probabilidades, Teste DM, Backtests, Regimes de Volatilidade, Limitações/Colapso, Síntese)
2. **Substituição**: Seção Optuna/top-10 → Calibração de Probabilidades (sem CSV Optuna; uso de proba_mean/proba_std)
3. **Tabelas**: Ajustadas para 4 folds (não 5)
4. **Colunas removidas**: Bal.Acc e Brier (dados não disponíveis para CNN-LSTM); F1-Score como substituto
5. **Abordagem honesta**: Colapso de classe (MCC=0 em 10/12 folds), DM sem significância, backtests mistos
6. **Seção 5.8 Limitações**: Colapso descrito, causas possíveis, o que não é invalidado

### Dados chave reportados
- **CNN-LSTM**: hit rate 50,76–51,55% (marginal sobre 50%)
- **Baselines**: 48,46–51,21%
- **DM direcional**: nenhum p < 0,05 (sem significância)
- **DM Brier**: todos p < 0,001 (CNN-LSTM melhor por natureza probabilística)
- **Backtests**: PETR4 long/short +4,10% (concentrado fold 1), VALE3 e ITUB4 ~0%

### Justificativa
- Abordagem honesta sobre resultados neutros/negativos é contribuição válida
- Mostra que deep learning não supera baselines simples em dados intradiários 15 min com validação rigorosa

### Arquivos modificados
- `TCC UFG/tex/05-resultados-discussao.tex`
- `TCC UFG/tex/06-consideracoes-finais.tex`
- `images/dm_heatmap_pvalores.png`

### Documentação
- [Capítulos 5 e 6](implementacoes/capitulos_05_06_resultados_2026_02_16.md)

---

## 2026-01-27 - Limpeza de Arquivos Obsoletos

### Contexto
- Projeto acumulou vários arquivos duplicados e obsoletos durante desenvolvimento
- Necessidade de organizar estrutura para facilitar manutenção
- Documentos consolidados substituem versões antigas

### Arquivos Removidos

1. **PROXIMOS_PASSOS.md**
   - Substituído por `PROXIMOS_PASSOS_CONSOLIDADO.md`
   - Versão consolidada mais completa e atualizada
   - Inclui cronograma e prioridades claras

2. **ESTRATEGIA_PROXIMOS_PASSOS.md**
   - Informações consolidadas em `PROXIMOS_PASSOS_CONSOLIDADO.md`
   - Recomendações já implementadas ou documentadas

3. **scripts/treinar_folds_4_5.sh**
   - Script específico para folds 4 e 5 não mais necessário
   - Funcionalidade coberta por `retreinar_completo.sh` e `treinar_outros_ativos.sh`
   - Treinamento completo agora é padrão

### Arquivos Mantidos (com justificativa)

- **ANALISE_MELHORIAS.md**, **MELHORIAS_IMPLEMENTADAS.md**, **RESUMO_MELHORIAS.md**, **GUIA_MELHORIAS.md**, **TESTE_RAPIDO.md**: mantidos (desde 2026-02-03 em [historico/projeto/](historico/README.md))

**Justificativa:** Cada documento tem propósito específico diferente:
- Análise: diagnóstico de problemas
- Implementadas: registro técnico detalhado
- Resumo: visão executiva
- Guia: instruções práticas
- Teste Rápido: workflow específico

### Impacto
- ✅ Estrutura mais limpa e organizada
- ✅ Redução de confusão sobre qual documento consultar
- ✅ Documentação consolidada facilita manutenção
- ✅ Scripts obsoletos removidos evitam uso incorreto

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
- [Correções do Treinamento](historico/implementacoes/correcoes_treinamento_2026_01_23.md) - Documentação completa (arquivado)

### Próximos Passos
- Avaliar resultados completos dos 5 folds
- Ajustar espaço de busca do Optuna (remover lr=0.01)
- Testar outras arquiteturas se necessário

---

## 2026-01-23 (tarde) - Implementação de Melhorias Técnicas

### Contexto
- Treinamento completo finalizado (2 horas)
- Resultados: Acurácia média 52.51%, F1=0.626, MCC=0.039
- Modelos não foram salvos (perda de 2 horas de trabalho)
- Consulta ao TCC para identificar melhorias possíveis

### Melhorias Implementadas

1. **Salvamento automático de modelos**
   - ModelCheckpoint para cada fold
   - Salva melhor modelo baseado em val_loss
   - Estrutura: `models/{ativo}/{modelo_tipo}/fold_{n}_checkpoint.keras`

2. **Gradient clipping**
   - Implementado com `clipnorm=1.0`
   - Previne explosão de gradientes
   - Conforme TCC Seção 4.4

3. **Otimizador AdamW**
   - Já estava implementado (confirmado)
   - Weight decay desacoplado
   - Melhor regularização que Adam

4. **Callbacks otimizados**
   - EarlyStopping (patience=10)
   - ReduceLROnPlateau (patience=5)
   - ModelCheckpoint (novo)

### Análise dos Resultados Atuais

**Walk-Forward (5 folds):**

| Fold | Acurácia | F1 | MCC | Neutros |
|------|----------|----|----|---------|
| 1 | 46.87% | 0.638 | 0.000 | 36.0% |
| 2 | 52.45% | 0.559 | 0.050 | 33.7% |
| 3 | 52.09% | 0.638 | 0.051 | 43.7% |
| 4 | 54.34% | 0.569 | 0.093 | 52.1% |
| 5 | 56.82% | 0.725 | 0.000 | 49.9% |
| **Média** | **52.51%** | **0.626** | **0.039** | **43.1%** |

**Interpretação:**
- ✅ Acurácia acima de baseline (~50%)
- ✅ Melhoria progressiva (46.87% → 56.82%)
- ⚠️ MCC muito baixo (sinal fraco)
- ⚠️ Alta variabilidade entre folds

### Técnicas Ainda Não Implementadas (Do TCC)

**Curto prazo:**
- Cosine annealing scheduler
- One-cycle scheduler
- Features adicionais (amplitude, volume)

**Médio prazo:**
- Ensemble de modelos (voting)
- Metaclassificador
- Retreinamento no maior prefixo

### Arquivos Modificados
- `src/models/cnn_lstm_model.py` - Gradient clipping
- `src/models/lstm_model.py` - Gradient clipping
- `src/train.py` - Salvamento de modelos
- `src/utils/optuna_optimizer.py` - Gradient clipping

### Documentação
- [Melhorias Técnicas](historico/implementacoes/melhorias_tecnicas_2026_01_23.md) - Documentação completa (arquivado)
- [Mudanças Completas](historico/implementacoes/mudancas_completas_2026_01_23_24.md) - Documentação completa (arquivado)

### Scripts Criados
- `src/scripts/analisar_modelos_salvos.py` - Análise de modelos salvos
- `src/scripts/ver_historico_epochs.py` - Visualização de epochs
- `src/scripts/teste_rapido_validacao.py` - Validação de testes rápidos
- `treinar_e_desligar.sh` - Treinamento com desligamento automático

### Status Atual
- ✅ Implementações completas
- ✅ 3 de 5 modelos salvos (folds 1-3)
- ❌ Faltam folds 4 e 5 (melhores resultados: 54.34% e 56.82%)
- ⏳ Próximo: Retreinar para salvar todos os modelos

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

## 2026-01-27 - Melhorias Críticas: Cosine Scheduler e Class Weights

### Contexto
- Análise dos resultados de treinamento dos modelos CNN-LSTM para PETR4, ITUB4 e VALE3
- Identificados problemas críticos: F1=0.0 e MCC=0.0 em alguns folds (PETR4 folds 2 e 3)
- Modelos prevendo sempre a mesma classe (colapso de aprendizado)

### Problemas Identificados

1. **F1=0.0 e MCC=0.0 em alguns folds**
   - PETR4 Folds 2 e 3 apresentavam F1=0.0 e MCC=0.0
   - Modelo prevendo sempre a mesma classe (provavelmente sempre "baixa")
   - Indica que modelo não está aprendendo padrões reais, apenas explorando desbalanceamento

2. **Acurácias abaixo do esperado**
   - PETR4 Fold 3: 47.15% (abaixo do baseline de 50%)
   - ITUB4 Fold 5: 50.00% (exatamente no acaso)

3. **Falta de técnicas do TCC**
   - Cosine Annealing Scheduler mencionado no TCC Seção 4.4 não implementado
   - Class weights usando cálculo manual (não robusto)

### Melhorias Implementadas

1. **Cosine Annealing Scheduler (TCC Seção 4.4)**
   - Implementado `CosineDecayRestarts` do TensorFlow
   - Reduz learning rate seguindo curva cosseno com restarts periódicos
   - Melhora convergência e pode aumentar acurácia em 1-3%
   - Arquivos modificados: `src/train.py`, `src/utils/optuna_optimizer.py`

2. **Class Weights Melhorados (sklearn)**
   - Substituído cálculo manual por `sklearn.utils.class_weight.compute_class_weight`
   - Estratégia 'balanced' mais robusta
   - Previne colapso para mesma classe
   - Arquivos modificados: `src/train.py`, `src/utils/optuna_optimizer.py`

3. **Monitoramento de Distribuição**
   - Adicionado aviso quando modelo prevê sempre mesma classe
   - Log detalhado durante otimização com distribuição de previsões
   - Facilita identificação de problemas durante treinamento
   - Arquivo modificado: `src/utils/optuna_optimizer.py`

### Resultados Esperados

**Antes das melhorias:**
- PETR4: 50.57% (com F1=0.0 em folds 2 e 3)
- ITUB4: 52.27%
- VALE3: 53.31%

**Depois das melhorias (esperado):**
- PETR4: 52-54% (sem F1=0.0)
- ITUB4: 54-56%
- VALE3: 55-57%

### Impacto Esperado
- ✅ Eliminação de F1=0.0 e MCC=0.0
- ✅ Acurácias mais consistentes entre folds
- ✅ Melhor convergência dos modelos
- ✅ Todas as técnicas principais da TCC Seção 4.4 implementadas

### Arquivos Modificados
- `src/train.py` - Adicionado cosine scheduler e class weights melhorados
- `src/utils/optuna_optimizer.py` - Mesmas melhorias para otimização com Optuna

### Documentação Criada
- `src/documentacao/historico/implementacoes/melhorias_criticas_2026_01_27.md` - Documentação completa das melhorias (arquivado)

---
