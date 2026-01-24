# Pipeline TCC - Predição Automática de Indicativos Financeiros para B3

Pipeline de predição automática de indicativos financeiros para Bolsa de Valores Considerando o Aspecto Temporal.

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

## Estrutura do Projeto

```
pipeline/
├── src/
│   ├── config.py              # Configurações globais
│   ├── data_processing/        # Pré-processamento e engenharia de features
│   ├── models/                 # Modelos baseline e principais
│   └── utils/                  # Métricas e validação walk-forward
├── data/
│   ├── raw/                    # Dados brutos (CSV com OHLCV)
│   └── processed/             # Dados processados e resultados
└── testar_baselines_walkforward.py  # Script de teste dos baselines
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

## Autor

Rafael da Silva Melo

## Licença

Este projeto é parte de um Trabalho de Conclusão de Curso (TCC).
