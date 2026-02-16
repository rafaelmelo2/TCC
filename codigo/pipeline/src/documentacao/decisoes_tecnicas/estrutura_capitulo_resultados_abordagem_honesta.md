# Decisão Técnica: Estrutura do Capítulo 5 e Abordagem Honesta sobre Resultados

**Data:** 2026-02-16  
**Tipo:** Estruturação do relatório TCC / Redação acadêmica  
**Status:** Implementado

---

## Contexto

- Capítulo 5 (Resultados e Discussão) é o mais importante do PFC2
- Dados reais do pipeline: CNN-LSTM com desempenho marginal (~50–51%), colapso de classe, DM sem significância
- Necessidade de reportar resultados honestamente sem invalidar metodologia

---

## Decisões Tomadas

### 1. Estrutura em 9 seções (não 8)
- Visão Geral → Comparação Geral → Estabilidade por Fold → Calibração de Probabilidades → Teste DM → Backtests → Regimes de Volatilidade → **Limitações e Colapso** → Síntese
- Seção dedicada a Limitações (5.8) em vez de mencionar superficialmente

### 2. Substituição Optuna/top-10 por Calibração
- Sem CSV com configurações Optuna disponível
- Seção substituída por análise de proba_mean/proba_std
- Mantém valor analítico sem depender de dados inexistentes

### 3. Tabelas com 4 folds (não 5)
- Volume efetivo dos dados e configuração de treino geraram 4 folds
- Template original previa 5; ajustado à realidade do dataset

### 4. Colunas removidas da tabela comparativa
- Bal.Acc e Brier removidas (dados não disponíveis para CNN-LSTM)
- F1-Score mantido como métrica de discriminação

### 5. Abordagem honesta
- Colapso de classe (MCC=0 em 10/12 folds) explicitamente reportado
- DM direcional: nenhum p < 0,05 reportado como “sem significância”
- Backtests mistos (PETR4 +4,1%; VALE3/ITUB4 ~0%) sem maquiagem
- DM Brier (p < 0,001): reportado como evidência de melhor calibração probabilística, não de superioridade direcional

---

## Justificativa

- Resultados neutros/negativos são contribuição válida à literatura
- Mostra que deep learning não supera baselines em dados intradiários 15 min com validação rigorosa
- Transparência fortalece credibilidade acadêmica
- Metodologia (walk-forward, DM, baselines) permanece válida; os resultados não invalidam o protocolo

---

## Impacto

- Capítulo 5 completo (~632 linhas) com dados reais
- Capítulo 6 reescrito para refletir conclusões honestas
- Trabalho defensável perante banca; contribuição científica explícita
