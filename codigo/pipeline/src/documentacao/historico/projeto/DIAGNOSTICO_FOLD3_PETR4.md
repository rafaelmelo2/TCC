# Diagnóstico: Fold 3 PETR4 - Modelo Colapsando

**Data:** 2026-01-27  
**Status:** Referência (diagnóstico colapso de classe)  
**Problema:** F1=0.0, MCC=0.0, modelo prevê sempre classe -1 (baixa)

---

## 1. Evidências

### Durante Otimização (Optuna):
```
Trial 0:  Pred=[1:0, -1:781], std=0.004 ⚠️ SEMPRE MESMA CLASSE
Trial 5:  Pred=[1:11, -1:770], std=0.004 (quase colapsou)
Trial 10: Pred=[1:0, -1:781], std=0.002 ⚠️ SEMPRE MESMA CLASSE
Trial 15: Pred=[1:0, -1:781], std=0.000 ⚠️ TOTALMENTE COLAPSADO
```

### Class Weights Aplicados:
```python
{0: 1.0098669114272603, 1: 0.9903240324032403}
```
→ **Quase iguais!** Classes estão balanceadas no treino.

### Resultado Final:
- Acurácia: 47.15% (pior que baseline de 50%)
- F1-Score: 0.0
- MCC: 0.0

---

## 2. Análise

### O que as melhorias NÃO resolveram:
1. ✅ Cosine Annealing Scheduler → Implementado mas não ajudou
2. ✅ Class weights (sklearn) → Calculados corretamente mas são quase iguais
3. ✅ Monitoramento → Funcionou, detectou o problema

### Causa Raiz:
**O problema NÃO é técnico (class weights, scheduler), é do PERÍODO:**

1. **Período extremamente difícil**: Fold 3 pode ter comportamento de mercado anômalo
2. **Features sem poder preditivo**: Indicadores técnicos não funcionam neste período
3. **Modelo não encontra padrões**: Converge para "sempre baixa" como melhor estratégia

### Por que o modelo prevê sempre "baixa"?
- Durante treino, aprende que no **conjunto de teste** deste fold, a classe majoritária é "baixa"
- Como não encontra padrões reais, usa esta estratégia trivial
- Isso dá ~47% porque no teste há ~47% de baixas

---

## 3. Soluções Propostas

### 1. Focal Loss (CRÍTICO) 🔴
**O que é:** Loss que penaliza mais erros em exemplos difíceis

**Por que ajuda:** 
- Força modelo a aprender ambas as classes
- Não permite colapso para uma classe

**Implementação:**
```python
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(K.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = alpha_t * K.pow((1 - p_t), gamma)
        loss = -focal_weight * K.log(p_t)
        return K.mean(loss)
    return focal_loss_fixed

# Usar na compilação:
model.compile(loss=focal_loss(gamma=2.0), ...)
```

**Benefício esperado:** Elimina F1=0.0, melhora +3-5%

### 2. Aumento Manual de Peso da Classe Minoritária 🟡
**O que fazer:**
- Multiplicar peso da classe minoritária por 2-3x
- Forçar modelo a dar mais atenção à classe que está ignorando

**Implementação:**
```python
if len(np.unique(y_train_binary)) > 1:
    weights = compute_class_weight('balanced', classes=classes, y=y_train_binary)
    # AUMENTAR peso da classe minoritária
    class_weight = {
        0: float(weights[0]) * 2.0,  # Dobrar peso de "baixa"
        1: float(weights[1]) * 2.0   # Dobrar peso de "alta"
    }
```

### 3. Early Stopping por Distribuição 🟡
**O que fazer:**
- Durante treinamento, verificar se modelo prevê ambas as classes
- Parar se 95%+ das previsões são da mesma classe

**Implementação:**
```python
class DistributionCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(X_val, verbose=0)
        n_high = np.sum(y_pred > 0.5)
        n_low = np.sum(y_pred <= 0.5)
        
        if n_high == 0 or n_low == 0:
            print(f"\n⚠️ Modelo colapsou! Parando treino...")
            self.model.stop_training = True
```

### 4. Features Temporais Adicionais 🟢
**O que adicionar:**
- Hora do dia (sin/cos encoding)
- Dia da semana
- Distância da abertura/fechamento
- Indicador de volatilidade extrema

**Benefício:** +2-5% acurácia (mas não resolve fold 3)

### 5. Data Augmentation (SMOTE/ADASYN) 🟢
**O que fazer:**
- Gerar exemplos sintéticos da classe minoritária
- Apenas para folds problemáticos

**Implementação:**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_filtered.reshape(len(X_train_filtered), -1),
    y_train_binary
)
```

---

## 4. Plano de Ação Recomendado

### Prioridade ALTA (Fazer AGORA):
1. **Implementar Focal Loss** ← Mais importante
2. Testar em fold 3 do PETR4

### Prioridade MÉDIA (Fazer depois):
3. Aumentar weight da classe minoritária manualmente
4. Implementar early stopping por distribuição
5. Adicionar features temporais

### Prioridade BAIXA:
6. SMOTE/data augmentation (complexo, pode causar overfitting)

---

## 5. Expectativa Realista

### Com Focal Loss:
- Fold 3 PETR4: 47.15% → **50-52%** (esperado)
- F1-Score: 0.0 → **0.3-0.5**
- MCC: 0.0 → **0.05-0.15**

**IMPORTANTE:** Fold 3 pode ser genuinamente difícil. Mesmo com todas as técnicas, pode não superar 52-53%.

### Literatura:
- Nem todos os períodos são previsíveis
- Alguns folds terão performance próxima do acaso
- Isso é **normal e aceitável** em finanças quantitativas

---

## 6. Notas Técnicas

### Por que Focal Loss funciona?
- Binary crossentropy trata todos os exemplos igualmente
- Focal Loss foca em exemplos difíceis e mal classificados
- Força modelo a não ignorar classe minoritária

### Quando usar?
- Problemas com classes desbalanceadas
- Modelos que colapsam para uma classe
- Quando class weights não são suficientes

### Referências:
- Lin et al. (2017): "Focal Loss for Dense Object Detection"
- Usado em computer vision, mas aplicável a séries temporais
