# Implementação: Engenharia de Features

**Data:** 2025-01-23  
**Status:** Implementado

---

## Features Implementadas

### 1. Retornos Logarítmicos
- Fórmula: `rt = ln(Pt) - ln(Pt-1)`
- Propriedade: aditividade temporal
- Uso: base para outras features e target

### 2. Médias Móveis Exponenciais (EMA)
- Períodos: 9, 21, 50 barras
- Uso: capturar tendências em diferentes escalas
- Cálculo: `ewm(span=periodo, adjust=False).mean()`

### 3. Relative Strength Index (RSI)
- Períodos: 9, 21, 50 barras
- Faixa: 0-100
- Uso: identificar sobrecompra/sobrevenda
- Cálculo: baseado em ganhos/perdas médios

### 4. Bandas de Bollinger
- Período: 20 barras
- Desvios: 2 desvios-padrão
- Features derivadas:
  - bb_upper, bb_lower, bb_middle
  - bb_width (largura normalizada)
  - bb_position (posição relativa do preço)

### 5. Volatilidade Realizada
- Período: 20 barras
- Cálculo: desvio-padrão dos retornos
- Uso: medir incerteza/risco

### 6. Target de Direção
- Valores: 1 (alta), -1 (baixa), 0 (neutro apenas se zero)
- Baseado em: retorno do próximo período
- Sem banda morta (threshold = 0.0)

---

## Justificativa das Features

### Múltiplos Períodos
- 9: curto prazo (intradiário)
- 21: médio prazo (1 dia)
- 50: longo prazo (2 dias)

### Features Técnicas Padrão
- Amplamente usadas em análise técnica
- Comprovadas na literatura
- Interpretáveis

---

## Distribuição Final dos Targets

- Alta (1): 47.1%
- Baixa (-1): 48.3%
- Neutro (0): 4.6%

**Observação:** Distribuição balanceada após remoção da banda morta

---

## Referências para TCC

### Seção: Metodologia - Engenharia de Features

**Pontos a mencionar:**
- Lista completa de features implementadas
- Justificativa dos períodos escolhidos
- Features derivadas (Bollinger)
- Target binário para classificação

---

## Arquivos

- `src/data_processing/feature_engineering.py`
