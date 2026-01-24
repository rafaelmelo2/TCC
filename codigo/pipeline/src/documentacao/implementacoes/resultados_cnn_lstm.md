# Resultados CNN-LSTM - Walk-Forward Validation

**Data do Teste:** 2025-01-26  
**Ativo:** VALE3  
**Modelo:** CNN-LSTM Híbrido  
**Configuração:** Hiperparâmetros padrão (sem otimização Optuna)

---

## Resultados por Fold

| Fold | Accuracy Direcional | Accuracy | F1-Score | MCC | Amostras Teste |
|------|---------------------|----------|----------|-----|----------------|
| 1 | **54.83%** | 54.83% | 0.229 | 0.051 | 546 |
| 2 | 52.12% | 52.12% | 0.252 | 0.030 | 546 |
| 3 | 53.65% | 53.65% | 0.429 | 0.065 | 546 |
| 4 | 48.23% | 48.23% | 0.093 | -0.041 | 546 |
| 5 | **40.52%** | 40.52% | 0.188 | -0.161 | 184 |

**Média Final:**
- Acurácia Direcional: **49.87%**
- Acurácia: 49.87%
- F1-Score: 0.238
- MCC: -0.011

---

## Comparação com Baselines

| Modelo | Accuracy Direcional | Diferença vs CNN-LSTM |
|--------|---------------------|----------------------|
| **Naive** | 50.50% | +0.63% |
| **Prophet** | 50.50% | +0.63% |
| **Drift** | 49.76% | -0.11% |
| **CNN-LSTM** | **49.87%** | - |
| **ARIMA** | 48.36% | -1.51% |

---

## Análise dos Resultados

### 1. Performance Geral
- **CNN-LSTM: 49.87%** vs **Baselines: 50.50%** (Naive/Prophet)
- O modelo CNN-LSTM está **ligeiramente pior** que os baselines mais simples
- Diferença: **-0.63%** (marginal, mas não supera)

### 2. Degradação ao Longo dos Folds
Há uma **degradação clara** de performance:
- Fold 1: 54.83% (melhor)
- Fold 2: 52.12%
- Fold 3: 53.65%
- Fold 4: 48.23%
- Fold 5: 40.52% (pior)

**Possíveis causas:**
- **Mudança de regime de mercado**: Padrões aprendidos nos primeiros folds não se aplicam aos últimos
- **Overfitting**: Modelo decorou padrões específicos do início dos dados
- **Fold 5 com poucos dados**: Apenas 184 amostras (vs 546 nos outros folds) - menos confiável

### 3. Análise por Métrica

**MCC (Matthews Correlation Coefficient):**
- Fold 1-3: MCC positivo (0.051, 0.030, 0.065) → Modelo melhor que aleatório
- Fold 4-5: MCC negativo (-0.041, -0.161) → Modelo **pior que aleatório**
- MCC médio: -0.011 → Indica que o modelo está praticamente no nível aleatório

**F1-Score:**
- Fold 3: 0.429 (melhor)
- Fold 4: 0.093 (pior)
- Alta variabilidade entre folds

### 4. Comparação com Expectativas

**Esperado:**
- CNN-LSTM deveria superar baselines simples (52-55%)
- Modelo híbrido deveria capturar padrões não-lineares

**Obtido:**
- 49.87% (praticamente igual aos baselines)
- Não supera Naive/Prophet (50.50%)

---

## Interpretação

### Por que o modelo não superou os baselines?

1. **Hiperparâmetros não otimizados**
   - Usamos valores padrão (não otimizados com Optuna)
   - Otimização pode melhorar significativamente

2. **Complexidade vs. Dados**
   - Modelo complexo (24.651 parâmetros) pode estar overfitting
   - Dados podem não ter padrões suficientemente fortes para justificar complexidade

3. **Degradação temporal**
   - Performance cai ao longo do tempo
   - Indica que padrões mudam (mudança de regime)

4. **Fold 5 problemático**
   - Apenas 184 amostras (vs 546 nos outros)
   - Menos confiável estatisticamente

### O que isso significa?

**Não é um resultado "ruim", mas sim um resultado importante:**
- ✅ Valida que o pipeline está funcionando corretamente
- ✅ Confirma que predição de direção é um problema difícil
- ✅ Mostra que modelos complexos não garantem melhor performance
- ⚠️ Indica necessidade de otimização de hiperparâmetros
- ⚠️ Sugere que pode haver overfitting

---

## Próximos Passos Recomendados

### 1. Otimização com Optuna (PRIORIDADE ALTA)
```bash
uv run python src/train.py --ativo VALE3 --modelo cnn_lstm --optuna --n-trials 30
```
- Otimizar hiperparâmetros pode melhorar 2-3%
- Esperado: 52-53% após otimização

### 2. Análise do Fold 5
- Investigar por que Fold 5 tem apenas 184 amostras
- Verificar se há problema na divisão walk-forward
- Considerar remover Fold 5 da análise se for muito pequeno

### 3. Testar LSTM Puro
```bash
uv run python src/train.py --ativo VALE3 --modelo lstm
```
- Comparar se LSTM puro é melhor que CNN-LSTM
- Pode indicar que CNN não está ajudando

### 4. Regularização
- Aumentar dropout (0.2 → 0.3 ou 0.4)
- Adicionar L2 regularization
- Reduzir complexidade do modelo

### 5. Análise de Features
- Verificar se todas as 12 features estão sendo usadas corretamente
- Testar com menos features (pode reduzir overfitting)

---

## Conclusão

**Resultado atual: 49.87% (ligeiramente abaixo dos baselines)**

**Status:** ⚠️ **Precisa de otimização**

O modelo está funcionando, mas não está superando baselines simples. Isso é comum em problemas de predição financeira e indica que:

1. **Otimização de hiperparâmetros é essencial** (próximo passo)
2. **Modelos complexos não garantem melhor performance** (importante para TCC)
3. **Degradação temporal precisa ser investigada** (mudança de regime?)

**Recomendação:** Executar otimização com Optuna antes de concluir que o modelo não funciona.

---

## Referências para TCC

### Seção: Resultados - Modelo CNN-LSTM

**Pontos a mencionar:**
- Resultado inicial: 49.87% (sem otimização)
- Comparação com baselines: Ligeiramente inferior a Naive/Prophet (50.50%)
- Degradação temporal observada (54.8% → 40.5%)
- Necessidade de otimização de hiperparâmetros
- Resultado após otimização: [a ser preenchido]

**Justificativa:**
- Resultado inicial serve como baseline do modelo
- Otimização deve melhorar performance
- Degradação temporal é um achado importante (mudança de regime)
