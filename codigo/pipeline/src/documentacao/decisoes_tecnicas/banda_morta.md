# Decisão Técnica: Remoção da Banda Morta

**Data:** 2025-01-23  
**Tipo:** Decisão de metodologia  
**Status:** Implementado

---

## Contexto Inicial

- Banda morta original implementada: ±0.0005 (0.05%)
- Objetivo: ignorar microvariações não significativas para trading
- Usada em: criação de target, cálculo de métricas, baselines

---

## Problema Identificado

### Análise dos Dados (VALE3, barras de 15min)

**Estatísticas dos Retornos:**
- Total: 35,153 retornos
- Média: 0.000012
- Mediana: 0.000000
- Desvio-padrão: 0.003443
- Mínimo: -0.042151
- Máximo: 0.078799
- Percentil 1%: -0.009389
- Percentil 99%: 0.009964

**Impacto da Banda Morta:**
- Retornos dentro da banda morta: 7,848 (22.3%)
- Retornos realmente zero: 1,624 (4.6%)
- **Amostras perdidas: 6,225 (17.7%)**

**Distribuição dos Targets:**
- Com banda morta: Alta 38.2%, Baixa 39.5%, Neutro 22.3%
- Sem banda morta: Alta 47.1%, Baixa 48.3%, Neutro 4.6%

---

## Análise Realizada

### Por que a banda morta não faz sentido aqui?

1. **Retornos intradiários são naturalmente pequenos**
   - Média: 0.000012 (0.0012%)
   - A maioria dos retornos está na ordem de 0.0001% a 0.01%
   - Banda morta de 0.05% elimina a maioria dos movimentos reais

2. **Para previsão de direção, qualquer movimento é relevante**
   - Objetivo: prever se preço sobe ou desce
   - Não importa a magnitude, apenas a direção
   - Microvariações ainda indicam direção

3. **Perda significativa de dados**
   - 17.7% dos dados úteis sendo descartados
   - Reduz tamanho do conjunto de treino/teste
   - Impacta capacidade de generalização

4. **Problema específico com ARIMA**
   - Forecasts muito pequenos (ordem de 0.000001)
   - Todos dentro da banda morta
   - Resultado: 100% de previsões neutras (F1=0.0)

---

## Decisão Tomada

### Remover banda morta completamente

**Mudanças implementadas:**

1. **Target Creation** (`feature_engineering.py`)
   - Threshold: 0.0 (apenas valores exatamente zero são neutros)
   - `retorno > 0` → 1 (alta)
   - `retorno < 0` → -1 (baixa)
   - `retorno == 0` → 0 (neutro)

2. **Métricas** (`metrics.py`)
   - Ignorar apenas valores exatamente zero
   - Usar todos os outros dados para cálculo

3. **Baselines** (`baselines.py`)
   - Todos usam apenas sinal (>0, <0, ==0)
   - Sem comparação com threshold

---

## Justificativa Técnica

### Por que essa decisão é correta?

1. **Natureza dos dados intradiários**
   - Retornos de 15 minutos são pequenos por natureza
   - Não são "ruído", são movimentos reais do mercado
   - Banda morta foi projetada para dados diários/semanais

2. **Objetivo do modelo**
   - Prever direção, não magnitude
   - Qualquer movimento indica direção
   - Microvariações são informação, não ruído

3. **Maximização de dados**
   - Mais dados = melhor treinamento
   - Especialmente importante para deep learning
   - Reduz overfitting

4. **Consistência metodológica**
   - Se vamos prever direção, devemos usar todas as direções
   - Banda morta introduz arbitrariedade desnecessária

---

## Impacto Mensurável

### Antes (com banda morta)
- Amostras utilizadas: 27,305 (77.7%)
- Amostras descartadas: 7,848 (22.3%)
- ARIMA F1_Score: 0.576
- ARIMA: 100% zeros nas previsões

### Depois (sem banda morta)
- Amostras utilizadas: 33,529 (95.4%)
- Amostras descartadas: 1,624 (4.6% - apenas zeros reais)
- ARIMA F1_Score: 0.593 (+2.9%)
- ARIMA: distribuição real (1=1637, -1=546, 0=205)

### Resultados dos Baselines

| Baseline | Acurácia | F1_Score | MCC |
|----------|----------|----------|-----|
| Naive    | 50.50%   | 0.315    | 0.002 |
| Drift    | 49.76%   | 0.543    | -0.002 |
| ARIMA    | 48.36%   | 0.593    | -0.029 |

---

## Referências para TCC

### Seção: Metodologia - Engenharia de Features

**Pontos a mencionar:**
- Decisão de não usar banda morta para dados intradiários
- Justificativa baseada em análise empírica
- Impacto na quantidade de dados disponíveis
- Comparação antes/depois

### Seção: Resultados - Baselines

**Pontos a mencionar:**
- Resultados dos baselines sem banda morta
- Distribuição de classes mais balanceada
- Impacto na qualidade das previsões (F1_Score)

---

## Arquivos Modificados

- `src/data_processing/feature_engineering.py`
- `src/utils/metrics.py`
- `src/models/baselines.py`
- `src/tests/testar_baselines_walkforward.py`

---

## Lições Aprendidas

- Banda morta é útil para dados de maior granularidade (diários)
- Para dados intradiários, pode eliminar informações importantes
- Sempre validar impacto empírico de decisões metodológicas
- Análise de distribuição dos dados é essencial antes de aplicar filtros
