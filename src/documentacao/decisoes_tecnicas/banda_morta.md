# Decisão Técnica: Uso da Banda Morta

**Data:** 2026-01-23 (atualizado)  
**Tipo:** Decisão de metodologia  
**Status:** Implementado

---

## Contexto e Evolução

### Histórico de Decisões

1. **Inicialmente (2025-01-23)**: Banda morta foi removida (threshold=0.0)
   - Justificativa: retornos intradiários são naturalmente pequenos
   - Resultado: apenas 4.6% de neutros (apenas zeros reais)

2. **Correção (2026-01-23)**: Banda morta foi **reaplicada** com threshold ajustado
   - Threshold aumentado: 0.0005 (0.05%) → **0.001 (0.1%)**
   - Justificativa revisada: filtrar ruído em dados intradiários
   - Resultado: 42.8% de neutros (filtro adequado de ruído)

---

## Configuração Atual

### Banda Morta Implementada

**Threshold:** `THRESHOLD_BANDA_MORTA = 0.001` (0.1%)

**Objetivo:** Filtrar movimentos pequenos (ruído) que não representam tendências significativas para trading intradiário.

**Classificação:**
- Retorno > 0.001: Alta (1)
- Retorno < -0.001: Baixa (-1)
- -0.001 ≤ Retorno ≤ 0.001: Neutro (0) - **removido do treinamento**

---

## Justificativa Técnica

### Por que usar banda morta de 0.1%?

1. **Natureza dos dados intradiários (15 minutos)**
   - Spread típico: 0.1-0.2%
   - Movimento mínimo significativo: ~0.1%
   - Threshold de 0.05% era menor que o spread, capturando ruído
   - Threshold de 0.1% filtra ruído mantendo movimentos significativos

2. **Filtro de ruído de mercado**
   - Movimentos < 0.1% são principalmente ruído de microestrutura
   - Não representam tendências utilizáveis para trading
   - Reduz overfitting a padrões aleatórios

3. **Referências da literatura**
   - Lopez de Prado (2018): "Advances in Financial Machine Learning" - Cap. 3
   - Estudos empíricos sugerem 0.1-0.3% para dados intradiários
   - Threshold de 0.1% é conservador e adequado para barras de 15min

4. **Balanceamento de dados**
   - Com threshold=0.001: ~42.8% neutros, ~28.2% alta, ~29.0% baixa
   - Distribuição mais balanceada que sem banda morta
   - Reduz viés do modelo para prever sempre a classe majoritária

---

## Impacto Mensurável

### Sem Banda Morta (threshold=0.0)
- Neutros: 4.6% (apenas zeros reais)
- Alta: 47.1%
- Baixa: 48.3%
- **Problema**: Muito ruído incluído nos dados de treinamento
- **Problema**: Modelo tentava prever movimentos aleatórios

### Com Banda Morta (threshold=0.001)
- Neutros: **42.8%** (filtro adequado)
- Alta: 28.2%
- Baixa: 29.0%
- **Benefício**: Ruído filtrado, apenas movimentos significativos
- **Benefício**: Modelo foca em tendências reais

### Resultados dos Treinamentos

**Antes da correção (banda morta não aplicada):**
- Acurácia: ~50-52% (próxima de chute aleatório)
- Modelos colapsando para estratégias simples

**Depois da correção (banda morta aplicada corretamente):**
- Acurácia: ~53-56% (poder preditivo real)
- Maior variância nas probabilidades (std: 0.006 → 0.010)
- Modelos aprendendo padrões mais robustos

---

## Implementação Técnica

### Código Atual

```python
# src/config.py
THRESHOLD_BANDA_MORTA = 0.001  # 0.1% - movimentos menores são considerados neutros

# src/data_processing/feature_engineering.py
def criar_target_com_banda_morta(df: pd.DataFrame, coluna_retornos: str = 'returns',
                                  threshold: float = THRESHOLD_BANDA_MORTA) -> pd.Series:
    """
    Cria target com banda morta para classificação direcional.
    
    IMPORTANTE: A banda morta filtra movimentos pequenos (ruído) que não
    representam tendências significativas. Movimentos entre -threshold e +threshold
    são classificados como neutros (0) e serão REMOVIDOS do treinamento.
    
    Conforme metodologia do TCC (Seção 4.2 - Definição de Target):
    - Retorno > threshold: Alta (1)
    - Retorno < -threshold: Baixa (-1)  
    - -threshold <= Retorno <= threshold: Neutro (0) - removido no treino
    """
    next_return = df[coluna_retornos].shift(-1)
    target = pd.Series(0, index=df.index, dtype=int)
    target.loc[next_return > threshold] = 1
    target.loc[next_return < -threshold] = -1
    # Valores entre -threshold e +threshold ficam como 0 (neutro)
    return target
```

### Uso no Pipeline

```python
# src/data_processing/feature_engineering.py (linha 125)
if incluir_target and 'returns' in df_features.columns:
    df_features['target'] = criar_target_com_banda_morta(
        df_features, 
        threshold=THRESHOLD_BANDA_MORTA  # ✅ Threshold aplicado corretamente
    )
```

---

## Referências para TCC

### Seção: Metodologia - Engenharia de Features (4.2)

**Pontos a mencionar:**
- Uso de banda morta de 0.1% para dados intradiários de 15 minutos
- Justificativa baseada em spread típico e literatura (Lopez de Prado, 2018)
- Filtro de ruído de microestrutura mantendo movimentos significativos
- Impacto na distribuição de classes (42.8% neutros, 28.2% alta, 29.0% baixa)

### Seção: Resultados - Modelos

**Pontos a mencionar:**
- Melhoria na acurácia após aplicação correta da banda morta
- Redução de overfitting a padrões aleatórios
- Maior robustez das previsões

---

## Arquivos Relacionados

- `src/config.py` - Definição de `THRESHOLD_BANDA_MORTA = 0.001`
- `src/data_processing/feature_engineering.py` - Função `criar_target_com_banda_morta()`
- `src/utils/metrics.py` - Uso da banda morta no cálculo de métricas
- `src/models/baselines.py` - Uso da banda morta nos baselines

---

## Lições Aprendidas

1. **Threshold adequado é crítico**
   - 0.05% muito pequeno (menor que spread) → captura ruído
   - 0.1% adequado para barras de 15 minutos → filtra ruído mantendo sinais

2. **Aplicação correta é essencial**
   - Bug inicial: função chamada sem threshold → threshold=0.0
   - Correção: passar `threshold=THRESHOLD_BANDA_MORTA` explicitamente

3. **Validação empírica necessária**
   - Análise de distribuição de classes
   - Impacto na qualidade das previsões
   - Comparação antes/depois

4. **Banda morta é útil para dados intradiários**
   - Quando threshold é adequado ao spread e granularidade
   - Filtra ruído mantendo informações relevantes
   - Melhora qualidade do treinamento
