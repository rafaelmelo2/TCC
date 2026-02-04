# Pasta `data/` — O que é cada parte e onde foi gerado

Resumo para consulta rápida (ex.: reunião com professor).

---

## 1. `data/raw/`

**O que é:** Dados brutos intradiários (OHLCV) em barras de 15 minutos.

**Conteúdo típico:**
- `PETR4_M15_20200101_20251231.csv`
- `VALE3_M15_20200101_20251231.csv`
- `ITUB4_M15_20200101_20251231.csv`

**Onde foi gerado:**  
**Não há script automático no repositório.** Os CSVs foram obtidos **fora do pipeline** (ex.: exportados do **MetaTrader 5** ou outra fonte), com período real de **22/10/2020 a 22/10/2025** (5 anos). O nome do arquivo usa `20200101_20251231` por compatibilidade. Documentação: `src/documentacao/periodo_dados.md`.

**Quem usa:** `train.py`, `rodar_backtest.py`, `analisar_modelos_salvos.py`, `visualizar_features.py`, testes de baselines, sensibilidade e previsão em tempo real — todos leem daqui.

---

## 2. `data/processed/`

**O que é:** Resultados de treinamento, validação walk-forward, baselines e testes estatísticos (CSVs e alguns PNGs).

**Principais arquivos e onde são gerados:**

| Arquivo / padrão | Onde é gerado |
|------------------|----------------|
| `{ATIVO}_baselines_walkforward.csv` (PETR4, VALE3, ITUB4) | `src/tests/testar_baselines_walkforward.py` |
| `{ATIVO}_cnn_lstm_walkforward.csv` | `src/train.py` (treino CNN-LSTM com walk-forward) |
| `{ATIVO}_cnn_lstm_analise_modelos.csv` | `src/scripts/analisar_modelos_salvos.py` |
| `comparativo_cnn_lstm_vs_baselines.csv` | `src/scripts/comparar_modelos.py` (consolida baselines + CNN-LSTM) |
| `testes_diebold_mariano.csv` | `src/scripts/rodar_testes_estatisticos.py` |
| `dm_resumo_pvalores.csv`, `dm_diferenca_perda_geral.csv`, `dm_heatmap_pvalores.png` | `src/scripts/gerar_tabelas_graficos_dm.py` (lê o CSV de DM acima) |

**Subpasta `data/processed/walkforward/`:**  
Arquivos de **sensibilidade** de walk-forward (ex.: `{ATIVO}_sensibilidade_walkforward_*.csv`), gerados por `src/tests/testar_sensibilidade_walkforward.py`.

---

## 3. `data/backtest/`

**O que é:** Resultados de backtest com custos (estratégias long-only e long/short) por ativo e fold.

**Onde foi gerado:**  
`src/scripts/rodar_backtest.py`.

**Conteúdo típico:**
- Um CSV por execução: `{ATIVO}_fold{N}_long_only_*.csv` ou `*_long_short_*.csv` (data/hora no nome).
- `historico_backtest.csv`: histórico consolidado (append a cada execução).

Detalhes (colunas, estratégias) estão em `data/backtest/README.md`.

---

## 4. `data/visualizacoes/`

**O que é:** Gráficos das features e do target, por ativo e por ano (EMA, Bollinger, RSI, retornos, volatilidade, target, correlações, distribuições, etc.).

**Onde foi gerado:**  
`src/scripts/visualizar_features.py`.

**Estrutura:**  
`data/visualizacoes/{ATIVO}/{ANO}/` — por exemplo `EMA_2022.png`, `RSI_2022.png`, `bollinger_2022.png`, `target_2022.png`, entre outros. Alinhado à engenharia de features do TCC (Seção 4.2).

---

## Resumo rápido para falar na reunião

- **raw:** Dados brutos 15 min (OHLCV); origem externa (ex.: MetaTrader 5), não gerados por script do repo.
- **processed:** Resultados de treino (walk-forward), baselines, comparação de modelos e testes Diebold-Mariano; gerados por `train.py`, `testar_baselines_walkforward.py`, `analisar_modelos_salvos.py`, `comparar_modelos.py`, `rodar_testes_estatisticos.py`, `gerar_tabelas_graficos_dm.py` e `testar_sensibilidade_walkforward.py`.
- **backtest:** Backtests com custos; gerados por `rodar_backtest.py`.
- **visualizacoes:** Gráficos de features/indicadores por ativo/ano; gerados por `visualizar_features.py`.

Todos os caminhos são relativos à raiz do pipeline (`codigo/pipeline/`). Para rodar da raiz do pipeline: `uv run python src/scripts/...` ou `uv run python src/train.py ...`.
