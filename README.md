# Pipeline TCC - Predição Automática de Indicativos Financeiros para B3

Pipeline de predição automática de indicativos financeiros para Bolsa de Valores Considerando o Aspecto Temporal.

**Documentação:** Toda a documentação técnica (implementações, decisões, cronograma, melhorias) está em [src/documentacao/](src/documentacao/) — índice em [INDICE.md](src/documentacao/INDICE.md). Comandos para rodar o pipeline: [COMANDOS_RODAR_TUDO.md](COMANDOS_RODAR_TUDO.md). Resumo para reunião com professor: [RESUMO_TCC_REUNIAO_PROFESSOR.md](RESUMO_TCC_REUNIAO_PROFESSOR.md).

## Pré-requisitos: dados em `data/raw/`

O pipeline espera CSVs de OHLCV em `data/raw/` com nomes no formato `{ATIVO}_M15_20201022_20251022.csv` (ex.: PETR4, VALE3, ITUB4). Colunas obrigatórias: `data`, `abertura`, `maxima`, `minima`, `fechamento`, `volume_real`. Sem esses arquivos, os scripts falham ao carregar dados. Detalhes em [data/README.md](data/README.md).

## Descrição

Este projeto implementa um modelo híbrido CNN+LSTM para prever a direção de movimentos de preços intradiários (barras de 15 minutos) em ações líquidas da B3.

## Instalação

### Usando uv (recomendado)

```bash
uv sync
```

### Usando pip

```bash
pip install -r requirements.txt
```

## Onde executar

Todos os comandos devem ser executados **a partir da raiz deste repositório** (a pasta onde está o README, `src/`, `data/`).

## Estrutura do Projeto

```
.
├── src/
│   ├── config.py               # Configurações globais
│   ├── data_processing/        # Pré-processamento e engenharia de features
│   ├── models/                 # Modelos baseline e principais
│   ├── scripts/                # Análise, comparativo, DM, backtest
│   ├── tests/                  # Baselines e sensibilidade
│   └── utils/                  # Métricas e validação walk-forward
├── scripts/                    # Shell scripts (ex.: rodar_todos_backtests.sh)
├── data/
│   ├── raw/                    # Dados brutos (CSV OHLCV); ver Pré-requisitos
│   ├── processed/              # Resultados de treino e testes
│   └── backtest/               # Resultados de backtest
├── README.md
├── COMANDOS_RODAR_TUDO.md
└── pyproject.toml
```

## Uso Rápido

### Testar baselines com walk-forward validation

```bash
uv run python src/tests/testar_baselines_walkforward.py
```

### Treinar modelo de deep learning

```bash
# Treinar modelo CNN-LSTM (padrão)
uv run python src/train.py --ativo VALE3 --modelo cnn_lstm

# Treinar modelo LSTM
uv run python src/train.py --ativo VALE3 --modelo lstm --epochs 100

# Com opções personalizadas
uv run python src/train.py --ativo PETR4 --modelo cnn_lstm --epochs 50 --batch-size 64
```

## Dependências Principais

- Python >= 3.10
- pandas >= 2.0.0
- numpy >= 1.24.0
- tensorflow >= 2.14.0
- scikit-learn >= 1.3.0
- statsmodels >= 0.14.0

## Metodologia

Conforme TCC - Capítulo 4:
- Validação walk-forward com embargo temporal
- Engenharia de features técnicas (EMA, RSI, Bollinger, etc.)
- Modelos baseline: Naive, Drift, ARIMA, Prophet
- Modelo principal: CNN-LSTM híbrido

## Compartilhamento

Este repositório contém apenas o pipeline (código + documentação). Quem clona precisa colocar os CSVs em `data/raw/` (formato no README) e seguir o [COMANDOS_RODAR_TUDO.md](COMANDOS_RODAR_TUDO.md). O `.gitignore` evita versionar dados brutos, `.venv` e logs.

## Autor

Rafael da Silva Melo

## Licença

Este projeto é parte de um Trabalho de Conclusão de Curso (TCC).
