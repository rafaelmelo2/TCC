# Resultados de Backtest (TCC Seção 4.5.1)

**Data:** 2026-02-02  
**Status:** Referência (estrutura dos arquivos gerados por `rodar_backtest.py`)

Esta pasta armazena os resultados das execuções do backtest com custos sobre previsões CNN-LSTM (walk-forward).

---

## 1. Estrutura dos arquivos

Os arquivos são gerados automaticamente pelo script `src/scripts/rodar_backtest.py`.

### 1.1 Nome dos arquivos

- **Backtest simples (uma execução):**  
  `{ATIVO}_fold{FOLD}_{ESTRATEGIA}_{AAAAMMDD}_{HHMMSS}.csv`  
  Exemplo: `VALE3_fold1_long_short_20260202_182309.csv`

- **Backtest com sensibilidade a custos:**  
  Além do arquivo acima, é gerado:  
  `{ATIVO}_fold{FOLD}_{ESTRATEGIA}_sensibilidade_{AAAAMMDD}_{HHMMSS}.csv`  
  Exemplo: `VALE3_fold1_long_short_sensibilidade_20260202_182309.csv`

- **Histórico consolidado:**  
  `historico_backtest.csv` — uma linha por execução (append), para comparar runs ao longo do tempo.

### 1.2 Colunas (resumo por execução)

| Coluna | Descrição |
|--------|-----------|
| data_hora | Data e hora da execução (AAAAMMDD_HHMMSS) |
| ativo | PETR4, VALE3 ou ITUB4 |
| fold | Número do fold (1 a 5) |
| estrategia | long_only ou long_short |
| retorno_liquido | Retorno após custos (decimal) |
| sharpe_ratio | Sharpe anualizado |
| max_drawdown | Maior queda pico-vale (decimal) |
| profit_factor | Ganhos / \|perdas\| |
| turnover | Frequência de mudança de posição |
| n_trades | Número de operações (pernas) |
| custo_total_reais | Custo total em R$ |
| capital_inicial | Capital no início (R$) |
| capital_final | Capital no final (R$) |
| n_barras | Número de barras no período de teste |

## 2. Estratégias

- **long_short:** Long quando sinal alta, short quando sinal baixa, neutro quando sinal neutro.
- **long_only:** Apenas long ou neutro; sinal baixa vira neutro.
