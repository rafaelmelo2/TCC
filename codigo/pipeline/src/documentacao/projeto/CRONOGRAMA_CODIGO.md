# Cronograma de Desenvolvimento - Código

**Data:** 2026-01-22  
**Status:** Referência (checklist de implementação 22/01–20/02/2026)  
**Foco:** Implementação de código e funcionalidades

---

## 1. Progresso atual

### Concluído
- ✅ `load_data.py` - Carregamento e validação de dados
- ✅ `validate_data.py` - Auditoria completa
- ✅ `config.py` - Configurações globais
- ✅ `feature_engineering.py` - Indicadores técnicos
- ✅ `baselines.py` - Naive, Drift, ARIMA
- ✅ `metrics.py` - Métricas de avaliação

### Concluído (adicional)
- ✅ Walk-forward validation (`validation.py`)
- ✅ Scripts de treinamento (`train.py`, Optuna, 3 ativos × 5 folds)
- ✅ CNN-LSTM, backtesting, Diebold-Mariano, comparativo

---

## 2. Checklist de desenvolvimento

### Semana 1: Fundação (22-28 Jan)

#### Dia 1 (22/01) - Dados e Estrutura
- [x] `load_data.py` - Carregamento de dados
- [x] `validate_data.py` - Auditoria
- [x] `config.py` - Configurações globais

#### Dia 2 (23/01) - Features
- [x] `feature_engineering.py` - Indicadores técnicos
  - [x] Retornos logarítmicos
  - [x] EMAs (9, 21, 50)
  - [x] RSIs (9, 21, 50)
  - [x] Bandas de Bollinger
  - [x] Volatilidade
  - [x] Target com banda morta

#### Dia 3 (24/01) - Baselines
- [x] `baselines.py`
  - [x] NaiveBaseline
  - [x] DriftBaseline
  - [x] ARIMABaseline
- [x] `metrics.py` - Métricas de avaliação

#### Dia 4 (25/01) - Walk-Forward
- [x] `validation.py` - WalkForwardValidator ✅
- [x] Testar walk-forward nos baselines ✅

#### Dia 5 (26/01) - Prophet
- [x] `prophet_model.py` - Baseline Prophet ✅
- [x] Consolidar resultados baselines ✅

#### Dia 6-7 (27-28/01) - Preparação DL
- [x] Preparar dados para deep learning (`prepare_sequences.py`) ✅
- [x] Script de treinamento (`train.py`) ✅
- [x] Configuração walk-forward documentada (`config.py`) ✅
- [x] Script de análise de sensibilidade (`testar_sensibilidade_walkforward.py`) ✅
- [x] Setup Optuna ✅
- [x] Arquitetura LSTM básica (já implementada em train.py) ✅

---

### Semana 2: LSTM (29 Jan - 04 Fev)

#### Dia 8-9 (29-30/01) - LSTM
- [x] `lstm_model.py` - Arquitetura LSTM ✅
- [x] `train.py` - Script de treinamento ✅
- [x] Setup Optuna para otimização bayesiana ✅
- [x] Definir espaços de busca de hiperparâmetros (`config.py`) ✅
  - [x] LSTM: lstm_units [32, 50, 64], dropout [0.1, 0.2, 0.3], learning_rate [1e-4, 1e-3, 1e-2], batch_size [16, 32, 64] ✅
  - [x] CNN-LSTM: conv_filters [32, 64, 128], conv_kernel_size [2, 3], lstm_units [32, 50, 64], dropout [0.1, 0.2, 0.3], learning_rate [1e-4, 1e-3, 1e-2], batch_size [16, 32, 64] ✅

#### Dia 10-11 (31/01-01/02) - Otimização
- [x] Implementar otimização bayesiana (Optuna) dentro de cada fold walk-forward ✅
- [x] Otimização no conjunto de validação interno (não no teste) ✅
- [x] Módulo `optuna_optimizer.py` criado ✅
- [x] Integração com `train.py` (flag `--optuna`) ✅
- [x] Testar otimização com LSTM ✅
- [x] Testar otimização com CNN-LSTM ✅
- [x] Analisar resultados Optuna ✅
- [x] Walk-forward completo com modelos otimizados (3 ativos, 5 folds) ✅

#### Dia 12-13 (02-03/02) - CNN-LSTM
- [x] `cnn_lstm_model.py` - Arquitetura híbrida ✅
- [x] Otimização CNN-LSTM com Optuna ✅
- [x] Hiperparâmetros: conv_filters, conv_kernel_size, lstm_units, dropout, learning_rate, batch_size ✅

#### Dia 14 (04/02) - Consolidação
- [x] Comparar LSTM vs CNN-LSTM (`comparar_modelos.py`, comparativo_cnn_lstm_vs_baselines.csv) ✅
- [x] Preparar para Semana 3 ✅

---

### Semana 3: CNN-LSTM e Backtests (05-11 Fev)

#### Dia 15-16 (05-06/02) - CNN-LSTM
- [x] Finalizar otimização CNN-LSTM ✅
- [x] Walk-forward completo (PETR4, VALE3, ITUB4, 5 folds) ✅
- [x] Avaliação completa (`analisar_modelos_salvos.py`) ✅

#### Dia 17-18 (07-08/02) - Backtests
- [x] `backtesting.py` - Backtester com custos ✅
- [x] Backtests para todos os modelos (long_short, long_only, 30 runs) ✅
- [x] Análise de resultados (`historico_backtest.csv`) ✅

#### Dia 19-20 (09-10/02) - Análises
- [x] Testes em múltiplos ativos (PETR4, VALE3, ITUB4) ✅
- [x] Análise comparativa (`comparar_modelos.py`) ✅
- [x] Documentação ✅

#### Dia 21 (11/02) - Fechamento
- [x] Consolidação de resultados ✅
- [x] Preparação para testes estatísticos ✅

---

### Semana 4: Testes e Robustez (12-18 Fev)

#### Dia 22-23 (12-13/02) - Testes Estatísticos
- [x] `diebold_mariano.py` - Teste DM ✅
- [x] Comparações estatísticas (`rodar_testes_estatisticos.py`, `gerar_tabelas_graficos_dm.py`) ✅
- [x] Análise por regimes (`--regimes`, `--brier`) ✅

#### Dia 24-25 (14-15/02) - Sensibilidade
- [x] Análise de sensibilidade a janelas walk-forward (`testar_sensibilidade_walkforward.py`) ✅
  - [x] Configurações: mais_permissivo, principal, mais_conservador, mais_treino, embargo_dia ✅
- [x] Executar análise de sensibilidade completa (walk-forward por ativo) ✅
- [ ] Análise de sensibilidade a features
- [x] Análise de sensibilidade a custos (`rodar_backtest.py --sensibilidade`) ✅
- [ ] Análise de sensibilidade a hiperparâmetros (variações dos valores otimizados)

#### Dia 26-27 (16-17/02) - Consolidação
- [x] Consolidar todos os resultados (resultados_consolidados_2026_02_03.md) ✅
- [ ] Gerar visualizações (Fase 7)
- [x] Preparar dados para escrita (tabelas DM, comparativo, backtest) ✅

#### Dia 28 (18/02) - Finalização Código
- [ ] Revisão final do código
- [x] Documentação (INDICE, implementacoes, historico) ✅
- [ ] Entrega técnica

---

## 3. Próximas tarefas (Fase 7)

- [ ] Visualizações para o TCC (gráficos performance, calibração, comparação baselines)
- [ ] Revisão final do código e entrega técnica
- [ ] Opcional: sensibilidade a features, sensibilidade a hiperparâmetros

---

## 4. Estatísticas

- **Módulos principais:** concluídos (load_data, validate_data, config, feature_engineering, baselines, prophet_model, validation, metrics, prepare_sequences, train, lstm_model, cnn_lstm_model, optuna_optimizer, backtesting, diebold_mariano)
- **Pipeline 1–5:** concluído (baselines, análise CNN-LSTM, comparativo, testes DM, backtests)
- **Progresso:** ~95% (falta Fase 7: visualizações, consolidação final)
