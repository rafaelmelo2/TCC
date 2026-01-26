# Teste Rápido de Validação - Passo a Passo

**Data:** 2026-01-23  
**Objetivo:** Testar melhorias antes de rodar treinamento completo durante a noite

---

## 🎯 Estratégia

1. **Teste Rápido** (~30 min): 10 trials, poucos folds
2. **Análise**: Script automático decide se prosseguir
3. **Treinamento Completo** (à noite): 50 trials, 5 folds

---

## 📋 Passo 1: Executar Teste Rápido

```bash
cd ~/Arquivos/TCC/codigo/pipeline

# Teste rápido: 10 trials do Optuna
uv run python src/train.py \
    --ativo VALE3 \
    --modelo cnn_lstm \
    --optuna \
    --n-trials 10 \
    --epochs 100
```

**Tempo estimado:** 30-40 minutos  
**O que faz:** Treina 2 primeiros folds com 10 trials de otimização

---

## 📊 Passo 2: Analisar Resultados

### Opção A: Script Automático (RECOMENDADO)

```bash
# Após o teste terminar
uv run python src/scripts/teste_rapido_validacao.py --ativo VALE3 --modelo cnn_lstm
```

**Saída esperada:**
```
==================================================
ANÁLISE DO TESTE RÁPIDO
==================================================

Resultados encontrados: 2 folds
   fold  accuracy_direcional  accuracy  f1_score    mcc
1     1            0.505000  0.505000  0.630000  0.055
2     2            0.535000  0.535000  0.590000  0.068

==================================================
CRITÉRIOS DE APROVAÇÃO
==================================================
  ✅ Acurácia média: 0.5200 > 0.51
  ✅ Fold 1: 0.5050 > 0.4687 (baseline)
  ✅ MCC médio: 0.0615 > 0.0390 (baseline)
  ✅ F1-Score médio: 0.6100 > 0.55
  ✅ Variabilidade: 0.0212 < 0.10 (estável)
  ✅ Modelos salvos: 2 arquivos

==================================================
✅ TESTE APROVADO - Pode prosseguir!
==================================================

Comando para treinamento completo (rodar à noite):

uv run python src/train.py \
    --ativo VALE3 \
    --modelo cnn_lstm \
    --optuna \
    --n-trials 50 \
    --epochs 150
```

### Opção B: Análise Manual

```bash
# Ver resultados
cat data/processed/VALE3_cnn_lstm_walkforward.csv

# Verificar modelos salvos
ls -lh models/VALE3/cnn_lstm/
```

---

## ✅ Critérios de Aprovação

O teste é **APROVADO** se:

| Critério | Valor Esperado | Baseline Atual |
|----------|----------------|----------------|
| Acurácia média | > 51% | 52.51% |
| Fold 1 | > 48% | 46.87% |
| MCC médio | > 0.04 | 0.039 |
| F1-Score | > 0.55 | 0.626 |
| Modelos salvos | 2 arquivos | - |

**Sinais de progressão:**
- ✅ Fold 1 melhor que antes (46.87%)
- ✅ MCC não é zero
- ✅ Trials convergindo sem NaN
- ✅ Previsões não totalmente desbalanceadas

---

## 🚀 Passo 3A: SE APROVADO - Rodar Completo

```bash
# Copiar e colar para rodar durante a noite
cd ~/Arquivos/TCC/codigo/pipeline

# Treinamento completo
nohup uv run python src/train.py \
    --ativo VALE3 \
    --modelo cnn_lstm \
    --optuna \
    --n-trials 50 \
    --epochs 150 \
    > logs/treinamento_completo_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Ver processo rodando
ps aux | grep train.py

# Acompanhar logs em tempo real
tail -f logs/treinamento_completo_*.log
```

**Tempo estimado:** 3-4 horas  
**Meta:** Acurácia 54-56%, MCC > 0.08

**Manhã seguinte:**
```bash
# Ver resultados
cat data/processed/VALE3_cnn_lstm_walkforward.csv

# Analisar modelos salvos
uv run python src/scripts/analisar_modelos_salvos.py --ativo VALE3 --modelo cnn_lstm
```

---

## 🔍 Passo 3B: SE REPROVADO - Investigar

### Problemas Possíveis

#### Problema 1: Fold 1 ainda muito baixo (< 48%)

**Possíveis causas:**
- Dados do Fold 1 têm regime diferente
- Hiperparâmetros não adequados
- Banda morta muito agressiva

**Soluções:**
```python
# Opção A: Aumentar threshold da banda morta
# Em config.py: THRESHOLD_BANDA_MORTA = 0.0015 (de 0.001)

# Opção B: Ajustar espaço de busca do Optuna
# Em optuna_optimizer.py: aumentar range de learning_rate
```

#### Problema 2: MCC = 0.0 persistente

**Possíveis causas:**
- Modelo prevendo sempre mesma classe
- Class weights não funcionando
- Validação interna desbalanceada

**Soluções:**
```python
# Verificar distribuição de previsões
# Ajustar class_weight no train.py
# Usar focal loss ao invés de binary crossentropy
```

#### Problema 3: Trials explodindo (NaN, loss infinito)

**Possíveis causas:**
- Learning rate muito alto
- Gradient clipping insuficiente
- Batch size muito pequeno

**Soluções:**
```python
# Opção A: Reduzir max learning rate
# Em optuna_optimizer.py: learning_rate = trial.suggest_float(..., 1e-4, 5e-3)

# Opção B: Aumentar gradient clipping
# Em cnn_lstm_model.py: gradient_clip_norm=0.5
```

#### Problema 4: Modelos não sendo salvos

**Possíveis causas:**
- Erro de permissão
- Diretório não criado
- Callback não configurado

**Soluções:**
```bash
# Criar diretório manualmente
mkdir -p models/VALE3/cnn_lstm

# Verificar permissões
ls -la models/
```

---

## 📝 Checklist de Execução

**Antes de dormir:**
- [ ] Executei teste rápido (10 trials)
- [ ] Analisei resultados com script
- [ ] Teste foi APROVADO
- [ ] Iniciei treinamento completo com `nohup`
- [ ] Verifiquei que processo está rodando
- [ ] Verifiquei logs iniciais (sem erros)

**Ao acordar:**
- [ ] Verificar se processo terminou (`ps aux | grep train`)
- [ ] Ver resultados finais (`cat data/processed/VALE3_cnn_lstm_walkforward.csv`)
- [ ] Analisar modelos salvos (script de análise)
- [ ] Comparar com baseline (52.51%)

---

## 🎯 Resultados Esperados

### Teste Rápido (2 folds)

| Métrica | Baseline | Meta Teste |
|---------|----------|------------|
| Acurácia Fold 1 | 46.87% | > 50% |
| Acurácia Fold 2 | 52.45% | > 53% |
| MCC médio | 0.039 | > 0.05 |

### Treinamento Completo (5 folds)

| Métrica | Baseline | Meta Completo |
|---------|----------|---------------|
| Acurácia média | 52.51% | 54-56% |
| MCC médio | 0.039 | > 0.08 |
| F1-Score | 0.626 | > 0.65 |
| Melhor fold | 56.82% | > 58% |

---

## 🛠️ Comandos de Emergência

### Se o treinamento travar

```bash
# Ver processos Python
ps aux | grep python

# Matar processo (substituir PID)
kill -9 <PID>

# Ver uso de GPU
nvidia-smi
```

### Se ficar sem espaço

```bash
# Ver espaço
df -h

# Limpar cache do TensorFlow
rm -rf ~/.keras/

# Limpar modelos antigos (CUIDADO!)
# rm -rf models/VALE3_old/
```

### Backup de segurança

```bash
# Antes de dormir, fazer backup dos modelos atuais
cp -r models/VALE3 models/VALE3_backup_$(date +%Y%m%d)
```

---

## 📞 Troubleshooting

### Erro: "Out of memory (GPU)"

```bash
# Reduzir batch size
uv run python src/train.py ... --batch-size 16  # (padrão: 32)
```

### Erro: "No module named 'optuna'"

```bash
# Reinstalar dependências
uv sync
```

### Erro: "Permission denied" ao salvar modelo

```bash
# Criar diretórios com permissões
mkdir -p models/VALE3/cnn_lstm
chmod -R 755 models/
```

---

## 📊 Monitoramento em Tempo Real

### Terminal 1: Logs

```bash
tail -f logs/treinamento_completo_*.log
```

### Terminal 2: GPU

```bash
watch -n 1 nvidia-smi
```

### Terminal 3: Resultados parciais

```bash
watch -n 10 "cat data/processed/VALE3_cnn_lstm_walkforward.csv"
```

---

## ✅ Conclusão

**Fluxo completo:**

1. **Agora:** Teste rápido (30 min)
2. **Analise:** Script automático
3. **Se OK:** Rodar completo durante a noite
4. **Manhã:** Verificar resultados e decidir próximos passos

**Meta final:** 54-56% de acurácia com modelos salvos!

---

**Última atualização:** 2026-01-23  
**Próximo passo:** Executar teste rápido
