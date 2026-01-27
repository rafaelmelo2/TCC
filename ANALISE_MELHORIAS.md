# Análise de Melhorias Necessárias

**Data:** 2026-01-27  
**Status:** Análise dos resultados e identificação de melhorias

---

## 🔴 Problemas Identificados

### 1. F1=0.0 e MCC=0.0 em alguns folds
**Problema:** PETR4 Folds 2 e 3 têm F1=0.0 e MCC=0.0
- **Causa:** Modelo prevendo sempre a mesma classe (provavelmente sempre "baixa")
- **Impacto:** Modelo não está aprendendo padrões reais, apenas explorando desbalanceamento

**Solução:**
- Melhorar class weights (usar sklearn.utils.class_weight)
- Adicionar monitoramento de distribuição de previsões
- Considerar focal loss ao invés de binary crossentropy

### 2. Acurácias baixas em alguns folds
- PETR4 Fold 3: 47.15% (abaixo do baseline)
- ITUB4 Fold 5: 50.00% (exatamente no acaso)

**Possíveis causas:**
- Períodos difíceis do mercado
- Modelo não convergindo adequadamente
- Features não informativas para esses períodos

---

## ✅ O que JÁ está implementado (do TCC)

1. ✅ Walk-forward validation (Seção 4.4)
2. ✅ Otimização bayesiana (Optuna) (Seção 4.4.2)
3. ✅ AdamW optimizer (Seção 4.4)
4. ✅ Gradient clipping (Seção 4.4)
5. ✅ Early stopping (Seção 4.4)
6. ✅ ReduceLROnPlateau (Seção 4.4)
7. ✅ Class weights básicos (Seção 4.4)
8. ✅ Banda morta (0.1%) (Seção 4.2)
9. ✅ Salvamento de modelos por fold

---

## ⏳ O que FALTA implementar (do TCC)

### 1. Cosine Annealing Scheduler (Seção 4.4) 🔴 CRÍTICO
**Status:** Não implementado  
**Benefício esperado:** +1-3% acurácia  
**Prioridade:** ALTA

**Implementação necessária:**
```python
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

# Cosine annealing com restarts
cosine_schedule = CosineDecayRestarts(
    initial_learning_rate=learning_rate,
    first_decay_steps=epochs // 2,
    t_mul=2.0,
    m_mul=1.0,
    alpha=1e-7
)
```

### 2. Melhorias em Class Weights 🔴 CRÍTICO
**Status:** Implementação básica (pode melhorar)  
**Problema:** Alguns folds ainda colapsam para mesma classe  
**Prioridade:** ALTA

**Melhorias:**
- Usar `sklearn.utils.class_weight.compute_class_weight`
- Adicionar monitoramento de distribuição de previsões
- Considerar focal loss para classes desbalanceadas

### 3. Features Adicionais (Seção 4.2) 🟡 MÉDIO
**Status:** Não implementado  
**Benefício esperado:** +2-5% acurácia  
**Prioridade:** MÉDIA

**Features a adicionar:**
- Amplitude high-low normalizada
- Variações de volume (volume_t / volume_ma)
- Hora do dia (sin/cos encoding)
- Indicador de fase do pregão (abertura/meio/fechamento)

### 4. Ensemble de Modelos (Seção 3.2) 🟡 MÉDIO
**Status:** Parcialmente implementado (script existe mas não integrado)  
**Benefício esperado:** +3-5% acurácia  
**Prioridade:** MÉDIA

**Implementação:**
- Voting dos 5 folds treinados
- Média ponderada de probabilidades
- Metaclassificador (opcional)

### 5. Retreinamento no Maior Prefixo (Seção 4.4) 🟢 BAIXO
**Status:** Não implementado  
**Benefício:** Modelo de produção mais robusto  
**Prioridade:** BAIXA

**Estratégia:**
- Após walk-forward, retreinar com TODOS os dados
- Usar melhores hiperparâmetros (média dos 5 folds)

---

## 📋 Plano de Ação Imediato

### Fase 1: Correções Críticas (HOJE)
1. ✅ Implementar Cosine Annealing Scheduler
2. ✅ Melhorar class weights (sklearn)
3. ✅ Adicionar monitoramento de distribuição de previsões
4. ✅ Testar em um fold problemático

### Fase 2: Melhorias Adicionais (PRÓXIMOS DIAS)
1. Adicionar features extras
2. Implementar ensemble
3. Retreinar modelos com melhorias

---

## 🎯 Resultados Esperados após Melhorias

**Atual:**
- VALE3: 53.31%
- PETR4: 50.57%
- ITUB4: 52.27%

**Após Fase 1 (Cosine + Class Weights):**
- Esperado: +2-4% → 55-57% média

**Após Fase 2 (Features + Ensemble):**
- Esperado: +5-8% → 58-62% média

---

## 📝 Notas Técnicas

### Por que F1=0.0 acontece?
Quando o modelo prevê sempre a mesma classe (ex: sempre "baixa"), temos:
- Precision = 0 (nenhum verdadeiro positivo)
- Recall = 0 (nenhum verdadeiro positivo)
- F1 = 2 * (0 * 0) / (0 + 0) = 0/0 = 0 (por definição)

### Como evitar?
1. Class weights mais agressivos
2. Focal loss (penaliza mais erros em classes minoritárias)
3. Oversampling/undersampling
4. Monitoramento durante treinamento
