# Guia de Melhorias - Como Aumentar a Acurácia

**Data:** 2026-01-23  
**Status:** Referência (como aumentar acurácia, técnicas TCC)

---

## 1. Sobre Aumentar Epochs

### Resposta Direta

**Com early stopping, aumentar epochs NÃO prejudica:**

- ✅ Para automaticamente quando não há melhoria (patience=10)
- ✅ Apenas define limite máximo de treinamento
- ✅ Útil para learning rates baixos que precisam de mais tempo
- ⏱️ Aumenta tempo máximo, mas para antes se convergir

**Configuração atual:**
```python
epochs = 100  # Máximo
patience = 10  # Para se 10 épocas sem melhoria
```

**Resultado típico:** Para entre 20-50 épocas

**Recomendação:** Aumentar para 150-200 epochs no próximo treino

---

## 2. Como Melhorar Acurácia (Seguindo TCC)

### TCC Seção 4.4 - Técnicas Recomendadas

✅ **JÁ IMPLEMENTADO:**
1. Gradient clipping (clipnorm=1.0)
2. AdamW optimizer
3. Early stopping
4. Reduce LR on plateau
5. Class weights (balanceamento)
6. Dropout regularization
7. **Salvamento automático de modelos** ← NOVO!

🔄 **PRÓXIMO A IMPLEMENTAR:**
1. Cosine annealing scheduler (TCC Seção 4.4)
2. Features adicionais (amplitude, volume)
3. Ensemble de modelos (TCC Seção 3.2)
4. Retreinamento no maior prefixo

---

## 3. IMPORTANTE: Modelos Salvos!

### Problema Resolvido

Antes: Treinamento de 2 horas sem salvar modelos  
Agora: **Salvamento automático a cada fold!**

### Onde estão os modelos?

```
models/
└── VALE3/
    └── cnn_lstm/
        ├── fold_1_checkpoint.keras  ← Melhor modelo do fold 1
        ├── fold_2_checkpoint.keras
        ├── fold_3_checkpoint.keras
        ├── fold_4_checkpoint.keras
        └── fold_5_checkpoint.keras
```

### Como usar os modelos salvos?

```python
from tensorflow import keras

# Carregar modelo do fold 5 (melhor: 56.82%)
model = keras.models.load_model('models/VALE3/cnn_lstm/fold_5_checkpoint.keras')

# Fazer previsões
predictions = model.predict(X_new)
directions = np.where(predictions > 0.5, 1, -1)
```

### Analisar modelos salvos

```bash
# Script criado para análise
uv run python src/scripts/analisar_modelos_salvos.py --ativo VALE3 --modelo cnn_lstm
```

---

## 4. Plano de Melhorias Sequencial

### Fase 1: Melhorias Imediatas (AGORA)

**O que fazer:**
1. ✅ Salvamento implementado
2. ✅ Gradient clipping implementado
3. Retreinar com configurações melhoradas

**Comando:**
```bash
# Treinar com mais trials e epochs
uv run python src/train.py \
    --ativo VALE3 \
    --modelo cnn_lstm \
    --optuna \
    --n-trials 50 \
    --epochs 150
```

**Tempo:** ~3-4 horas  
**Melhoria esperada:** 52.51% → 54-56%  
**Modelos salvos:** Sim, automaticamente!

### Fase 2: Features Adicionais (Depois)

**Implementar:**
- Amplitude high-low normalizada
- Variações de volume
- Hora do dia (sin/cos)
- Fase do pregão

**Tempo:** 1h implementação + 2h treino  
**Melhoria esperada:** +2-3% acurácia

### Fase 3: Ensemble (Depois)

**Estratégia:**
- Usar os 5 modelos salvos (um por fold)
- Voting ou média de probabilidades
- Pode chegar a 58-60% de acurácia

**Tempo:** ~30 min implementação  
**Melhoria esperada:** +3-5% acurácia

---

## 5. Melhorias Prioritárias

### Opção A: Retreinar com Melhorias Implementadas ⭐ RECOMENDADO

```bash
# Melhor custo-benefício
uv run python src/train.py \
    --ativo VALE3 \
    --modelo cnn_lstm \
    --optuna \
    --n-trials 50 \
    --epochs 150
```

**Por que:**
- Aproveita gradient clipping (novo)
- Mais trials = melhor hiperparâmetros
- Mais epochs = melhor convergência
- Modelos salvos automaticamente
- **Melhoria esperada: 54-56% acurácia**

### Opção B: Ensemble com Modelos Atuais

```python
# Usar os 5 modelos que você já tem
# Fazer média das previsões
# Melhoria esperada: 54-55% acurácia
```

**Por que:**
- Rápido (30 min)
- Não precisa retreinar
- Aproveita trabalho já feito

### Opção C: Implementar Schedulers + Retreinar

```python
# Adicionar cosine scheduler
# Retreinar tudo
# Melhoria esperada: 55-57% acurácia
```

**Por que:**
- Máximo de melhoria técnica
- Segue TCC rigorosamente
- Mais demorado (~1h implementação + 4h treino)

---

## 6. Expectativas Realistas

### Literatura de Previsão Intradiária

| Método | Acurácia Típica |
|--------|-----------------|
| Baseline (naive) | ~50% |
| ARIMA | 48-52% |
| LSTM single | 52-56% |
| CNN-LSTM | 54-58% |
| Ensemble | 56-62% |
| Estado da arte | 58-65% |

**Nosso resultado atual:**
- 52.51% com CNN-LSTM → **dentro do esperado**
- Com melhorias: 54-58% → **realista**
- Com ensemble: 56-60% → **otimista**

### Por que não 90%+?

1. **Mercado eficiente**: Se fosse fácil, todos fariam
2. **Ruído intradiário**: Movimentos de 15min são muito voláteis
3. **Limitação fundamental**: Preço futuro depende de fatores desconhecidos

**Mas 55% já é rentável!**
- Com boa gestão de risco
- Usando custos de transação
- Stop loss adequado

---

## 7. Comandos Rápidos

### Retreinar AGORA com Melhorias

```bash
cd ~/Arquivos/TCC/codigo/pipeline

# CNN-LSTM melhorado (RECOMENDADO)
uv run python src/train.py --ativo VALE3 --modelo cnn_lstm --optuna --n-trials 50 --epochs 150

# Verificar modelos salvos depois
ls -lh models/VALE3/cnn_lstm/

# Analisar modelos
uv run python src/scripts/analisar_modelos_salvos.py --ativo VALE3 --modelo cnn_lstm
```

### Treinar LSTM Puro (Baseline 3)

```bash
# Para comparação
uv run python src/train.py --ativo VALE3 --modelo lstm --optuna --n-trials 30 --epochs 150
```

### Treinar Outros Ativos

```bash
# PETR4
uv run python src/train.py --ativo PETR4 --modelo cnn_lstm --optuna --n-trials 30 --epochs 150

# ITUB4
uv run python src/train.py --ativo ITUB4 --modelo cnn_lstm --optuna --n-trials 30 --epochs 150
```

---

## 8. O Que Mudou

### Antes (Treinamento Anterior)
```
❌ Sem salvamento de modelos
❌ Sem gradient clipping
❌ Patience baixo (5)
❌ Poucas epochs (30)
→ Acurácia: ~52% mas modelos perdidos
```

### Agora (Com Melhorias)
```
✅ Salvamento automático por fold
✅ Gradient clipping (clipnorm=1.0)
✅ Patience adequado (10)
✅ Mais epochs (100-150)
✅ AdamW optimizer
→ Acurácia esperada: 54-56%
→ Modelos salvos e utilizáveis!
```

---

## 9. Resumo das Melhorias

### Técnicas do TCC Implementadas

| Técnica | Status | Impacto Esperado |
|---------|--------|------------------|
| Walk-forward | ✅ Implementado | Essencial |
| Optuna bayesiano | ✅ Implementado | +2-3% |
| AdamW | ✅ Implementado | +1% |
| Gradient clipping | ✅ Implementado | +0.5-1% |
| Early stopping | ✅ Implementado | Previne overfit |
| Dropout | ✅ Implementado | Regularização |
| Class weights | ✅ Implementado | Balanceamento |
| Salvamento | ✅ Implementado | Preserva trabalho |
| Cosine scheduler | ⏳ Próximo | +1-2% |
| Ensemble | ⏳ Próximo | +3-5% |
| Features extras | ⏳ Próximo | +2-3% |

**Total esperado:** 52.51% → 58-62% (com todas as técnicas)

---

## 10. Minha Recomendação

### Opção 1: Retreinar AGORA ⭐ MELHOR

```bash
uv run python src/train.py --ativo VALE3 --modelo cnn_lstm --optuna --n-trials 50 --epochs 150
```

**Por que:**
- Aproveita gradient clipping (NOVO)
- Modelos serão salvos (NOVO)
- Mais trials = melhores hiperparâmetros
- Tempo: 3-4 horas
- **Melhoria esperada: 54-56%**

### Opção 2: Implementar Schedulers DEPOIS Retreinar

**Etapas:**
1. Implementar cosine scheduler (~30 min)
2. Retreinar (~4 horas)
3. **Melhoria esperada: 55-58%**

### Opção 3: Ensemble com Modelos Atuais

**Se quiser resultado rápido:**
- Implementar voting dos 5 folds (~30 min)
- Não precisa retreinar
- **Melhoria esperada: 54-55%**

---

## 11. Documentação Completa

- [Melhorias Técnicas](../implementacoes/melhorias_tecnicas_2026_01_23.md)
- [Correções do Treinamento](../implementacoes/correcoes_treinamento_2026_01_23.md)

---

**O que você quer fazer agora?**

1. Retreinar com as melhorias (RECOMENDO)
2. Implementar ensemble rápido
3. Implementar schedulers antes de retreinar
