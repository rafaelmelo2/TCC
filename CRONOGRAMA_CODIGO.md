# 📅 CRONOGRAMA DE DESENVOLVIMENTO - CÓDIGO

**Foco**: Implementação de código e funcionalidades  
**Período**: 22 de Janeiro a 20 de Fevereiro de 2026

---

## ✅ PROGRESSO ATUAL

### Concluído
- ✅ `load_data.py` - Carregamento e validação de dados
- ✅ `validate_data.py` - Auditoria completa
- ✅ `config.py` - Configurações globais
- ✅ `feature_engineering.py` - Indicadores técnicos
- ✅ `baselines.py` - Naive, Drift, ARIMA
- ✅ `metrics.py` - Métricas de avaliação

### Em Andamento
- 🔄 Walk-forward validation
- 🔄 Scripts de treinamento

---

## 📋 CHECKLIST DE DESENVOLVIMENTO

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
- [ ] `validation.py` - WalkForwardValidator
- [ ] Testar walk-forward nos baselines

#### Dia 5 (26/01) - Prophet
- [ ] `prophet_model.py` - Baseline Prophet
- [ ] Consolidar resultados baselines

#### Dia 6-7 (27-28/01) - Preparação DL
- [ ] Preparar dados para deep learning
- [ ] Setup Optuna
- [ ] Arquitetura LSTM básica

---

### Semana 2: LSTM (29 Jan - 04 Fev)

#### Dia 8-9 (29-30/01) - LSTM
- [ ] `lstm_model.py` - Arquitetura LSTM
- [ ] `train.py` - Script de treinamento
- [ ] Otimização com Optuna

#### Dia 10-11 (31/01-01/02) - Otimização
- [ ] Analisar resultados Optuna
- [ ] Retreinar com melhores hiperparâmetros
- [ ] Walk-forward completo com LSTM

#### Dia 12-13 (02-03/02) - CNN-LSTM
- [ ] `cnn_lstm_model.py` - Arquitetura híbrida
- [ ] Otimização CNN-LSTM

#### Dia 14 (04/02) - Consolidação
- [ ] Comparar LSTM vs CNN-LSTM
- [ ] Preparar para Semana 3

---

### Semana 3: CNN-LSTM e Backtests (05-11 Fev)

#### Dia 15-16 (05-06/02) - CNN-LSTM
- [ ] Finalizar otimização CNN-LSTM
- [ ] Walk-forward completo
- [ ] Avaliação completa

#### Dia 17-18 (07-08/02) - Backtests
- [ ] `backtesting.py` - Backtester com custos
- [ ] Backtests para todos os modelos
- [ ] Análise de resultados

#### Dia 19-20 (09-10/02) - Análises
- [ ] Testes em múltiplos ativos
- [ ] Análise comparativa
- [ ] Documentação

#### Dia 21 (11/02) - Fechamento
- [ ] Consolidação de resultados
- [ ] Preparação para testes estatísticos

---

### Semana 4: Testes e Robustez (12-18 Fev)

#### Dia 22-23 (12-13/02) - Testes Estatísticos
- [ ] `diebold_mariano.py` - Teste DM
- [ ] Comparações estatísticas
- [ ] Análise por regimes

#### Dia 24-25 (14-15/02) - Sensibilidade
- [ ] Análise de sensibilidade a janelas
- [ ] Análise de sensibilidade a features
- [ ] Análise de sensibilidade a custos

#### Dia 26-27 (16-17/02) - Consolidação
- [ ] Consolidar todos os resultados
- [ ] Gerar visualizações
- [ ] Preparar dados para escrita

#### Dia 28 (18/02) - Finalização Código
- [ ] Revisão final do código
- [ ] Documentação
- [ ] Entrega técnica

---

## 🎯 PRÓXIMAS TAREFAS IMEDIATAS

1. **Walk-Forward Validation** (`validation.py`)
   - Implementar WalkForwardValidator
   - Testar com baselines

2. **Prophet Baseline** (`prophet_model.py`)
   - Implementar ProphetBaseline
   - Integrar com pipeline

3. **Preparação Deep Learning**
   - Criar sequências temporais
   - Setup de treinamento

---

## 📊 ESTATÍSTICAS

- **Módulos criados**: 6/15
- **Progresso**: ~40%
- **Próximo marco**: Walk-forward validation
