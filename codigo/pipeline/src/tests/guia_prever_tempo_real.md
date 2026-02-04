# Guia: Dados Necessários para Previsão em Tempo Real

**Data:** 2026-01-26  
**Status:** Referência operacional  
**Objetivo:** Explicar quais dados são necessários e como obtê-los em tempo real para previsão em tempo real.

---

## 1. Dados Necessários como Input

### Formato dos Dados

O modelo precisa das **últimas 60 barras de 15 minutos** (JANELA_TEMPORAL_STEPS) com as seguintes informações:

#### Colunas Obrigatórias (OHLCV):

```python
{
    'data': '2026-01-26 14:45:00',  # Timestamp da barra
    'abertura': 65.20,              # Preço de abertura (open)
    'maxima': 65.35,                # Preço máximo (high)
    'minima': 65.15,                # Preço mínimo (low)
    'fechamento': 65.30,            # Preço de fechamento (close)
    'volume_real': 1000000          # Volume negociado
}
```

### Quantidade Mínima

- **Mínimo:** 60 barras de 15 minutos
- **Equivale a:** ~15 horas de pregão (60 barras × 15 min = 900 min = 15h)
- **Recomendado:** Ter algumas barras extras para garantir

### Exemplo de Estrutura

```python
import pandas as pd

# Exemplo: últimas 60 barras de VALE3
dados = pd.DataFrame({
    'data': [
        '2026-01-26 10:00:00',
        '2026-01-26 10:15:00',
        '2026-01-26 10:30:00',
        # ... mais 57 barras ...
        '2026-01-26 14:45:00'  # Última barra fechada
    ],
    'abertura': [65.20, 65.30, 65.25, ...],
    'maxima': [65.35, 65.40, 65.30, ...],
    'minima': [65.15, 65.25, 65.20, ...],
    'fechamento': [65.30, 65.35, 65.28, ...],
    'volume_real': [1000000, 1200000, 950000, ...]
})

# Total: 60 linhas (barras)
```

---

## 2. Como Obter Dados em Tempo Real

### Opção 1: MetaTrader 5 (Recomendado)

**Vantagens:**
- ✅ Dados em tempo real
- ✅ API Python disponível
- ✅ Suporta múltiplos ativos
- ✅ Histórico completo

**Instalação:**
```bash
pip install MetaTrader5
```

**Código de exemplo:**
```python
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# Inicializar MT5
if not mt5.initialize():
    print("Erro ao inicializar MT5")
    mt5.shutdown()

# Obter últimas 60 barras de VALE3
ativo = "VALE3"
timeframe = mt5.TIMEFRAME_M15  # 15 minutos
num_barras = 60

# Copiar dados históricos
rates = mt5.copy_rates_from_pos(ativo, timeframe, 0, num_barras)

# Converter para DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# Renomear colunas para formato esperado
df = df.rename(columns={
    'time': 'data',
    'open': 'abertura',
    'high': 'maxima',
    'low': 'minima',
    'close': 'fechamento',
    'tick_volume': 'volume_real'  # ou 'real_volume' se disponível
})

# Selecionar apenas colunas necessárias
df = df[['data', 'abertura', 'maxima', 'minima', 'fechamento', 'volume_real']]

# Usar para previsão
# ... (passar df para script de previsão)
```

---

### Opção 2: API B3 (Brasil Bolsa Balcão)

**Vantagens:**
- ✅ Dados oficiais da B3
- ✅ Gratuito
- ✅ Dados históricos disponíveis

**Limitações:**
- ⚠️ Pode ter delay
- ⚠️ Requer parsing de formato específico

**Exemplo de uso:**
```python
import requests
import pandas as pd

# API B3 para cotação em tempo real
# (verificar documentação oficial da B3)
url = "https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/cotacoes/"
# ... implementar conforme API disponível
```

---

### Opção 3: Atualizar CSV Manualmente

**Quando usar:**
- Dados não estão em tempo real
- Testes iniciais
- Validação do modelo

**Como fazer:**
```python
import pandas as pd
from datetime import datetime

# Carregar CSV existente
df = pd.read_csv('data/raw/VALE3_M15_20200101_20251231.csv')

# Adicionar nova barra (exemplo)
nova_barra = {
    'data': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'abertura': 65.20,
    'maxima': 65.35,
    'minima': 65.15,
    'fechamento': 65.30,
    'volume_real': 1000000
}

# Adicionar ao DataFrame
df = pd.concat([df, pd.DataFrame([nova_barra])], ignore_index=True)

# Salvar
df.to_csv('data/raw/VALE3_M15_20200101_20251231.csv', index=False)
```

---

### Opção 4: Yahoo Finance / Alpha Vantage

**Vantagens:**
- ✅ Fácil de usar
- ✅ Múltiplos ativos

**Limitações:**
- ⚠️ Pode ter delay
- ⚠️ Limites de requisições

**Exemplo:**
```python
import yfinance as yf
import pandas as pd

# Obter dados de VALE3
ticker = yf.Ticker("VALE3.SA")

# Obter dados intradiários (15 minutos)
df = ticker.history(period="1d", interval="15m")

# Renomear colunas
df = df.reset_index()
df = df.rename(columns={
    'Datetime': 'data',
    'Open': 'abertura',
    'High': 'maxima',
    'Low': 'minima',
    'Close': 'fechamento',
    'Volume': 'volume_real'
})

# Selecionar últimas 60 barras
df = df.tail(60)
```

---

## 3. Script de Previsão em Tempo Real

### Versão Melhorada (aceita DataFrame diretamente)

```python
# prever_com_dados.py
import pandas as pd
from src.scripts.prever_tempo_real import prever_proxima_vela

# Obter dados (de qualquer fonte)
dados_reais = obter_dados_em_tempo_real()  # Sua função

# Fazer previsão
resultado = prever_proxima_vela(
    ativo='VALE3',
    dados_df=dados_reais,  # DataFrame com últimas 60 barras
    usar_ensemble=True
)

print(f"Direção: {resultado['direcao']}")
print(f"Probabilidade: {resultado['probabilidade']:.2%}")
```

---

## 4. Fluxo Completo de Previsão em Tempo Real

### 1. Obter Dados Atualizados

```python
# A cada 15 minutos (quando nova vela fecha):
dados = obter_ultimas_60_barras(ativo='VALE3')
```

### 2. Preparar Dados

```python
# O script faz automaticamente:
# - Cria features (EMA, RSI, Bollinger, etc.)
# - Normaliza dados
# - Cria sequência temporal (60 barras)
```

### 3. Fazer Previsão

```python
# Carrega modelo e prevê
resultado = prever_proxima_vela(dados_df=dados)
```

### 4. Interpretar Resultado

```python
if resultado['direcao'] == 'ALTA' and resultado['confianca'] == 'ALTA':
    # Sinal forte de alta
    print("Sinal de COMPRA")
elif resultado['direcao'] == 'BAIXA' and resultado['confianca'] == 'ALTA':
    # Sinal forte de baixa
    print("Sinal de VENDA")
else:
    # Sinal fraco, aguardar
    print("Aguardar confirmação")
```

---

## 5. Importante: Normalização

**PROBLEMA ATUAL:** O script cria um scaler novo a cada previsão.

**SOLUÇÃO IDEAL:** Usar o scaler do treino (será implementado).

**Por enquanto:** O script funciona, mas pode ter pequena perda de precisão.

---

## 6. Checklist para Previsão em Tempo Real

- [ ] Ter acesso a dados em tempo real (MT5, API, etc.)
- [ ] Obter últimas 60 barras de 15 minutos
- [ ] Verificar formato: data, abertura, maxima, minima, fechamento, volume_real
- [ ] Garantir que dados estão ordenados por tempo (mais antigo → mais recente)
- [ ] Executar script de previsão
- [ ] Interpretar resultado com gestão de risco

---

## 7. Resumo

**Dados necessários:**
- ✅ 60 barras de 15 minutos
- ✅ Colunas: data, abertura, maxima, minima, fechamento, volume_real
- ✅ Ordenadas por tempo (crescente)

**Como obter:**
1. MetaTrader 5 (recomendado)
2. API B3
3. Yahoo Finance
4. Atualização manual de CSV

**Próximo passo:** Implementar script que aceita DataFrame diretamente (não apenas CSV)

---

**Última atualização:** 2026-01-26
