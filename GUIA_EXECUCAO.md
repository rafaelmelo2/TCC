# Guia de Execução - TCC 2

## Passo a Passo para Rodar o Pipeline Completo

### 1. Treinamento Básico (CNN-LSTM)
```bash
cd codigo/pipeline/src
python train.py
```
**Resultado**: Modelo treinado salvo em `models/melhor_modelo_tcc.keras`

### 2. Otimização de Hiperparâmetros (Optuna)
```bash
python train_optuna.py
```
**Resultado**: Melhores hiperparâmetros em `models/melhores_hiperparametros.json` e modelo otimizado em `models/melhor_modelo_optuna.keras`

### 3. Backtest com Métricas Financeiras
```bash
python backtest.py
```
**Resultado**: Curva de capital em `models/backtest_result.png` + métricas (Sharpe, Turnover, Profit Factor)

### 4. Validação Walk-Forward (Comparação com Baselines)
Crie script `comparar_baselines.py` usando `WalkForwardValidator` de `utils/validation.py` para rodar Naive, ARIMA, Prophet, LSTM e CNN-LSTM sob mesmo protocolo.

### 5. Visualizações (Calibração, Importância, Equity)
Use funções de `utils/visualizer.py`:
- `plot_reliability_diagram()` para calibração (ECE)
- `plot_feature_importance()` para importância de features
- `plot_equity_curves()` para comparar estratégias

**Dependências**: Instalar `optuna`, `scipy` via `uv pip install -r requirements.txt`
