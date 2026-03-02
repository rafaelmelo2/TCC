# Pipeline TCC - Predição Automática de Indicativos Financeiros para B3

Pipeline de predição automática de indicativos financeiros para Bolsa de Valores Considerando o Aspecto Temporal.

**Documentação:** Toda a documentação técnica (implementações, decisões, cronograma, melhorias) está em [src/documentacao/](src/documentacao/) — índice em [INDICE.md](src/documentacao/INDICE.md). **Para rodar o pipeline completo (baselines, análise, comparativo, testes DM, backtests), use o guia:** [COMANDOS_RODAR_TUDO.md](COMANDOS_RODAR_TUDO.md). Resumo para reunião com professor: [RESUMO_TCC_REUNIAO_PROFESSOR.md](RESUMO_TCC_REUNIAO_PROFESSOR.md).

## Onde executar

**Sempre** a partir do diretório do pipeline:

```bash
cd codigo/pipeline   # ou o caminho onde está a pasta pipeline
```

Todos os comandos abaixo pressupõem que você está nesse diretório.

## Descrição

Este projeto implementa um modelo híbrido CNN+LSTM para prever a direção de movimentos de preços intradiários (barras de 15 minutos) em ações líquidas da B3.

## Pré-requisitos: dados em `data/raw/`

O pipeline espera CSVs de OHLCV em `data/raw/` com nomes no formato:

- `PETR4_M15_20201022_20251022.csv`
- `VALE3_M15_20201022_20251022.csv`
- `ITUB4_M15_20201022_20251022.csv`

Colunas obrigatórias: `data`, `abertura`, `maxima`, `minima`, `fechamento`, `volume_real`. Período dos dados: 22/10/2020 a 22/10/2025 (conforme `src/config.py`). Sem esses arquivos, os scripts de baselines, treino e backtest falham ao carregar dados.

## Instalação

### Usando uv (recomendado)

```bash
uv sync
```

### Usando pip

```bash
pip install -r requirements.txt
```

## Estrutura do Projeto

```
pipeline/
├── src/
│   ├── config.py               # Configurações globais
│   ├── data_processing/        # Pré-processamento e engenharia de features
│   ├── models/                 # Modelos baseline e principais
│   ├── scripts/                # Scripts de análise, comparativo, DM, backtest
│   ├── tests/                  # Testes de baselines e sensibilidade
│   └── utils/                  # Métricas e validação walk-forward
├── scripts/                    # Shell scripts (ex.: rodar_todos_backtests.sh)
├── data/
│   ├── raw/                    # Dados brutos (CSV OHLCV; nomes conforme acima)
│   ├── processed/              # Dados processados e resultados
│   └── backtest/               # Resultados de backtest
└── ...
```

## Uso Rápido

Recomendado ter os CSVs em `data/raw/` antes de rodar (veja [Pré-requisitos](#pré-requisitos-dados-em-dataraw)).

### Testar baselines com walk-forward validation

```bash
uv run python src/tests/testar_baselines_walkforward.py
# Ou os 3 ativos de uma vez:
uv run python src/tests/testar_baselines_walkforward.py --todos
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

### Rodar o pipeline completo (baselines → análise → comparativo → DM → backtests)

A ordem completa dos passos e todos os comandos estão em **[COMANDOS_RODAR_TUDO.md](COMANDOS_RODAR_TUDO.md)**. Resumo: (1) baselines walk-forward para os 3 ativos, (2) análise dos modelos CNN-LSTM salvos, (3) tabela comparativa, (4) testes Diebold-Mariano e gráficos, (5) backtests (long_short e long_only). O script em lote dos backtests: `./scripts/rodar_todos_backtests.sh` (a partir do diretório do pipeline).

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

## Compartilhamento (repositório reprodutível)

Para compartilhar o trabalho de forma reprodutível, use **apenas esta pasta** (`codigo/pipeline/`) como repositório. Ela é autocontida: README, dependências, código e documentação estão aqui; o texto do TCC e do artigo ficam de fora. Quem clona só precisa ter os CSVs em `data/raw/` (formato descrito em [Pré-requisitos](#pré-requisitos-dados-em-dataraw)) e rodar os comandos do [COMANDOS_RODAR_TUDO.md](COMANDOS_RODAR_TUDO.md). O `.gitignore` desta pasta evita versionar dados brutos, `.venv` e logs.

## Autor

Rafael da Silva Melo

## Licença

Este projeto é parte de um Trabalho de Conclusão de Curso (TCC).
