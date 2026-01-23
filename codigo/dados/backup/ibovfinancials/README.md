# Download de Dados da B3

Script simples para baixar dados históricos da B3 usando a API IbovFinancials.

## ⚠️ IMPORTANTE - Limite de Requisições

- **API permite apenas 90 dias por requisição**
- **Período 2020-2025 = ~22 requisições por ticker**
- **3 tickers × 22 períodos = ~66 requisições totais**
- **Você tem 200 requisições disponíveis**

O script calcula automaticamente quantas requisições serão necessárias e pede confirmação antes de executar.

## Tickers

- **PETR4** - Petrobras
- **VALE3** - Vale
- **ITUB4** - Itaú Unibanco

## Período Configurado

- **Início**: 1º de janeiro de 2020
- **Fim**: Data atual
- **Timeframe**: 15 minutos (candlesticks)
- **Delay entre requisições**: 2 segundos

## Como Usar

### 1. Instalar dependências

```bash
pip install requests pandas
```

### 2. Executar download

```bash
python baixar_dados_b3.py
```

O script irá:

1. Mostrar quantas requisições serão necessárias
2. Pedir confirmação
3. Baixar os dados em lotes de 90 dias
4. Salvar automaticamente em CSV e JSON

## Estrutura dos Dados

Os dados são salvos em:

```
dados/
├── csv/
│   ├── PETR4_15min_2020-01-01_2025-10-21_20251021_143022.csv
│   ├── VALE3_15min_2020-01-01_2025-10-21_20251021_143022.csv
│   └── ITUB4_15min_2020-01-01_2025-10-21_20251021_143022.csv
└── json/
    ├── PETR4_15min_2020-01-01_2025-10-21_20251021_143022.json
    ├── VALE3_15min_2020-01-01_2025-10-21_20251021_143022.json
    └── ITUB4_15min_2020-01-01_2025-10-21_20251021_143022.json
```

### Nome dos Arquivos

Formato: `{TICKER}_{TIMEFRAME}min_{START_DATE}_{END_DATE}_{TIMESTAMP}.csv`

### Colunas Esperadas

- `date` ou `datetime` - Data e hora
- `open` - Preço de abertura
- `high` - Preço máximo
- `low` - Preço mínimo
- `close` - Preço de fechamento
- `volume` - Volume negociado

## Configuração

Edite as variáveis no início do arquivo `baixar_dados_b3.py`:

```python
# Tickers para baixar
TICKERS = ["PETR4", "VALE3", "ITUB4"]

# Período (CUIDADO: mais dias = mais requisições!)
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Timeframe em minutos
TIMEFRAME = "15"

# Delay entre requisições (segundos)
DELAY_SECONDS = 2
```

## Economizar Requisições

Para economizar requisições, você pode:

1. **Baixar período menor**:

```python
START_DATE = "2024-01-01"  # Só 2024-2025 = ~7 requisições/ticker
```

2. **Baixar menos tickers**:

```python
TICKERS = ["PETR4"]  # Apenas 1 ticker
```

3. **Usar timeframe maior** (menos dados):

```python
TIMEFRAME = "60"  # 60 minutos ao invés de 15
```

## Token da API

Token já configurado no script:

```python
API_TOKEN = "f43158b7b639228278a0911ccb50ec720c6acf5c"
```
