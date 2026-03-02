# Próximos Passos Consolidados - TCC

**Data:** 2026-01-27  
**Status:** Cronograma e prioridades (prazo 20/02/2026)

---

## 1. Situação Atual do Projeto

### ✅ O que já foi concluído

#### Semana 1 (22-28 Jan) - COMPLETA ✅
- [x] Estrutura de dados e validação (`load_data.py`, `validate_data.py`)
- [x] Engenharia de features (`feature_engineering.py`)
- [x] Baselines implementados (Naive, Drift, ARIMA, Prophet)
- [x] Walk-forward validation (`validation.py`)
- [x] Métricas de avaliação (`metrics.py`)

#### Semana 2 (29 Jan-04 Fev) - EM ANDAMENTO 🔄
- [x] Arquitetura LSTM (`lstm_model.py`)
- [x] Arquitetura CNN-LSTM (`cnn_lstm_model.py`)
- [x] Script de treinamento (`train.py`)
- [x] Otimização bayesiana com Optuna (`optuna_optimizer.py`)
- [x] Treinamento completo dos 3 ativos (PETR4, VALE3, ITUB4)
- [x] Melhorias técnicas implementadas:
  - [x] Focal Loss (gamma=5.0, alpha=0.5)
  - [x] Class weights balanceados
  - [x] Cosine Annealing Scheduler
  - [x] Gradient clipping
  - [x] Monitoramento de colapso de modelo

### 📈 Resultados Atuais

#### Acurácia Direcional por Ativo (Walk-Forward)

| Ativo | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Média** |
|-------|--------|--------|--------|--------|--------|-----------|
| **VALE3** | 47.46% | 50.00% | 50.19% | 48.40% | 43.18% | **47.85%** ⚠️ |
| **PETR4** | 51.23% | 49.62% | 52.85% | 44.70% | 48.53% | **49.39%** ⚠️ |
| **ITUB4** | 51.90% | 49.84% | 52.98% | 55.56% | 45.59% | **51.17%** ✅ |
| **GERAL** | - | - | - | - | - | **49.47%** ⚠️ |

**Observações:**
- ⚠️ Média geral abaixo de 50% (baseline)
- ⚠️ VALE3 e PETR4 com performance abaixo do esperado
- ✅ ITUB4 acima de 50%, mas ainda marginal
- ⚠️ Alguns folds com F1=0.0 e MCC=0.0 (modelo colapsando)

---

## 2. PRÓXIMOS PASSOS PRIORITÁRIOS

### 🔴 PRIORIDADE CRÍTICA (Esta Semana - 27 Jan a 02 Fev)

#### 1. **Retreinar Modelos com Focal Loss** ⭐⭐⭐
**Status:** Pendente  
**Prazo:** 27-28 Jan  
**Tempo estimado:** 6-9 horas

**Ação:**
```bash
# A partir do diretório do pipeline
# Retreinar todos os ativos com focal loss (quando o script existir)
# ./retreinar_completo.sh
uv run python src/train.py --ativo PETR4 --modelo cnn_lstm
# ... idem VALE3, ITUB4
```

**O que esperar:**
- Melhoria de 2-5% na acurácia média
- Redução de colapsos (F1=0.0)
- Modelos mais robustos

**Critérios de sucesso:**
- ✅ Média geral > 51%
- ✅ Redução de folds com F1=0.0
- ✅ Todos os 3 ativos acima de 50%

---

#### 2. **Análise Comparativa com Baselines** ⭐⭐⭐
**Status:** Pendente  
**Prazo:** 29-30 Jan  
**Tempo estimado:** 2-3 horas

**Ação:**
```bash
# Treinar baselines para comparação
uv run python src/tests/testar_baselines_walkforward.py --ativo PETR4
uv run python src/tests/testar_baselines_walkforward.py --ativo VALE3
uv run python src/tests/testar_baselines_walkforward.py --ativo ITUB4

# Analisar resultados
uv run python src/scripts/analisar_modelos_salvos.py
```

**Entregáveis:**
- Tabela comparativa CNN-LSTM vs Baselines
- Análise estatística de significância
- Documentação para TCC

---

#### 3. **Implementar Backtesting com Custos** ⭐⭐
**Status:** Pendente  
**Prazo:** 31 Jan - 01 Fev  
**Tempo estimado:** 4-6 horas

**Conforme Cronograma:** Semana 3, Dia 17-18 (07-08 Fev)

**Ação:**
Criar `src/utils/backtesting.py` com:
- Simulação de trading com custos de transação
- Cálculo de métricas financeiras (Sharpe, Max Drawdown, Profit Factor)
- Análise de turnover

**Métricas a calcular:**
- Retorno líquido (após custos)
- Sharpe Ratio
- Maximum Drawdown
- Profit Factor
- Turnover (frequência de trades)

---

### 🟡 PRIORIDADE ALTA (Próxima Semana - 02-08 Fev)

#### 4. **Análise de Sensibilidade Completa** ⭐⭐
**Status:** Parcialmente implementado  
**Prazo:** 02-03 Fev  
**Tempo estimado:** 3-4 horas

**Ação:**
```bash
# Executar análise de sensibilidade já implementada
uv run python src/tests/testar_sensibilidade_walkforward.py --ativo PETR4
uv run python src/tests/testar_sensibilidade_walkforward.py --ativo VALE3
uv run python src/tests/testar_sensibilidade_walkforward.py --ativo ITUB4
```

**Análises pendentes:**
- [x] Sensibilidade a janelas walk-forward ✅
- [ ] Sensibilidade a features (remover/adicionar indicadores)
- [ ] Sensibilidade a custos de transação
- [ ] Sensibilidade a hiperparâmetros (variações dos valores otimizados)

---

#### 5. **Testes Estatísticos (Diebold-Mariano)** ⭐⭐
**Status:** Pendente  
**Prazo:** 04-05 Fev  
**Tempo estimado:** 4-5 horas

**Conforme Cronograma:** Semana 4, Dia 22-23 (12-13 Fev)

**Ação:**
Criar `src/utils/diebold_mariano.py` para:
- Comparar previsões de diferentes modelos estatisticamente
- Testar significância das diferenças de performance
- Análise por regimes de mercado (alta/baixa volatilidade)

---

#### 6. **Otimização CNN-LSTM com Optuna** ⭐
**Status:** Implementado, mas pode melhorar  
**Prazo:** 06-07 Fev  
**Tempo estimado:** 2-3 horas

**Ação:**
- Aumentar número de trials (de 20 para 50-100)
- Expandir espaço de busca de hiperparâmetros
- Testar diferentes arquiteturas (mais camadas, diferentes tamanhos)

---

### 🟢 PRIORIDADE MÉDIA (Semana 3-4 - 09-18 Fev)

#### 7. **Features Temporais Adicionais** ⭐
**Status:** Opcional  
**Prazo:** 09-10 Fev  
**Tempo estimado:** 2-3 horas

**Features a adicionar:**
- Hora do dia (encoding sin/cos)
- Dia da semana (one-hot)
- Distância da abertura/fechamento
- Volatilidade de curto prazo (janela 5-10)

**Arquivo:** `src/data_processing/feature_engineering.py`

---

#### 8. **Ensemble de Modelos** ⭐
**Status:** Opcional  
**Prazo:** 11-12 Fev  
**Tempo estimado:** 3-4 horas

**Estratégias:**
- Voting simples (média das probabilidades)
- Weighted average (pesos por acurácia)
- Stacking (metaclassificador)

**Criar:** `src/scripts/ensemble_models.py`

---

#### 9. **Visualizações e Gráficos para TCC** ⭐
**Status:** Pendente  
**Prazo:** 13-14 Fev  
**Tempo estimado:** 4-5 horas

**Gráficos necessários:**
- Performance walk-forward por fold
- Comparação com baselines
- Curvas de calibração (Brier Score)
- Confusion matrices por ativo
- Distribuição de retornos
- Análise de regimes de mercado

**Criar:** `src/scripts/gerar_graficos_tcc.py`

---

#### 10. **Consolidação de Resultados** ⭐
**Status:** Pendente  
**Prazo:** 15-16 Fev  
**Tempo estimado:** 3-4 horas

**Ações:**
- Consolidar todas as métricas em tabelas
- Calcular estatísticas descritivas
- Gerar relatório final
- Preparar dados para escrita do TCC

---

### 📝 PRIORIDADE BAIXA (Finalização - 17-20 Fev)

#### 11. **Documentação Final**
**Status:** Em andamento  
**Prazo:** 17-18 Fev  
**Tempo estimado:** 2-3 horas

**Documentos a atualizar:**
- README.md com instruções completas
- Documentação de código (docstrings)
- Guia de reprodução dos resultados

---

#### 12. **Revisão Final e Testes**
**Status:** Pendente  
**Prazo:** 19-20 Fev  
**Tempo estimado:** 2-3 horas

**Ações:**
- Testar reprodução completa em ambiente limpo
- Verificar consistência dos resultados
- Revisar código para bugs
- Preparar código para entrega

---

## 3. CHECKLIST SEMANAL

### Semana Atual (27 Jan - 02 Fev)

- [ ] **Dia 1 (27 Jan):** Retreinar modelos com focal loss
- [ ] **Dia 2 (28 Jan):** Analisar resultados do retreinamento
- [ ] **Dia 3 (29 Jan):** Comparar com baselines
- [ ] **Dia 4 (30 Jan):** Finalizar análise comparativa
- [ ] **Dia 5 (31 Jan):** Implementar backtesting (início)
- [ ] **Dia 6 (01 Fev):** Finalizar backtesting
- [ ] **Dia 7 (02 Fev):** Análise de sensibilidade

---

## 4. RISCOS E MITIGAÇÕES

### Risco 1: Performance abaixo de 50%
**Probabilidade:** Média  
**Impacto:** Alto  
**Mitigação:**
- Retreinar com focal loss (já em andamento)
- Adicionar features temporais
- Implementar ensemble

### Risco 2: Atraso no cronograma
**Probabilidade:** Média  
**Impacto:** Médio  
**Mitigação:**
- Priorizar tarefas críticas
- Deixar features opcionais para depois
- Focar em resultados reprodutíveis

### Risco 3: Problemas técnicos
**Probabilidade:** Baixa  
**Impacto:** Médio  
**Mitigação:**
- Manter backups dos modelos
- Documentar problemas encontrados
- Testar em ambiente limpo

---

## 5. MÉTRICAS DE SUCESSO

### Objetivos Mínimos (Para TCC)
- ✅ Acurácia média > 50% (superar baseline)
- ✅ Validação em 3 ativos diferentes
- ✅ Walk-forward validation rigorosa
- ✅ Comparação com baselines
- ✅ Métricas financeiras calculadas

### Objetivos Ideais
- 🎯 Acurácia média > 52-53%
- 🎯 Sharpe Ratio > 1.0
- 🎯 Resultados consistentes entre ativos
- 🎯 Testes estatísticos significativos

---

## 6. COMANDOS ÚTEIS

### Retreinar todos os ativos
```bash
# A partir do diretório do pipeline
uv run python src/train.py --ativo PETR4 --modelo cnn_lstm
# Repetir para VALE3 e ITUB4; ou usar script em lote se existir
```

### Analisar resultados
```bash
uv run python src/scripts/analisar_modelos_salvos.py
```

### Ver histórico de treinamento
```bash
uv run python src/scripts/ver_historico_epochs.py --ativo PETR4
```

### Testar baselines
```bash
uv run python src/tests/testar_baselines_walkforward.py --ativo PETR4
```

---

## 7. REFERÊNCIAS DO CRONOGRAMA

- **CRONOGRAMA_CODIGO.md:** Cronograma detalhado de desenvolvimento
- **CRONOGRAMA.md:** Cronograma geral do TCC
- **DIAGNOSTICO_FOLD3_PETR4.md:** Análise do problema identificado (em [historico/projeto/](../historico/README.md))
- **GUIA_MELHORIAS.md:** Guia prático de melhorias (em [historico/projeto/](../historico/README.md))

---

## 8. CONCLUSÃO

**Status Atual:** Projeto em fase de refinamento e análise  
**Próxima Ação Crítica:** Retreinar modelos com focal loss  
**Prazo:** 24 dias até entrega final  

**Foco Imediato:**
1. Melhorar performance dos modelos (retreinar)
2. Validar metodologia (comparar com baselines)
3. Implementar análises financeiras (backtesting)

**Mantra:** "Reprodutibilidade > Performance Perfeita"

---

**Última atualização:** 2026-01-27  
**Próxima revisão:** Após retreinamento completo
