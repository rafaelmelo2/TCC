# Estratégia - Próximos Passos do TCC

**Data:** 2026-01-26  
**Status Atual:** VALE3 completo (5 folds), PETR4 e ITUB4 pendentes

---

## 📊 Situação Atual

### VALE3 (Completo ✅)
- **Fold 4:** 54.34% acurácia direcional
- **Fold 5:** 52.27% acurácia direcional  
- **Média:** ~53.31%
- **Modelos:** Todos os 5 folds salvos

### PETR4 e ITUB4
- ❌ Ainda não treinados
- ✅ Dados disponíveis e prontos

---

## 🎯 Recomendação: Treinar Outras Ações PRIMEIRO

### Por que esta é a melhor estratégia?

#### 1. **Validação Metodológica (CRÍTICO para TCC)**
- TCC precisa demonstrar que modelo funciona em **múltiplos ativos**
- Resultados em apenas 1 ativo são insuficientes para conclusões robustas
- Literatura exige validação em pelo menos 3 ativos diferentes

#### 2. **Robustez Estatística**
- 3 ativos = 15 folds no total (5 por ativo)
- Amostra maior = conclusões mais confiáveis
- Permite análise de variabilidade entre ativos

#### 3. **Comparação e Análise**
- Identificar se modelo funciona melhor em certos tipos de ativos
- Comparar volatilidade, liquidez, setor
- Análise mais rica para o TCC

#### 4. **Tempo vs Benefício**
- Treinar outras ações: ~6-8 horas (2 ativos × 3-4h cada)
- Implementar melhorias: ~10-15 horas (código + testes + retreinar tudo)
- **Melhor custo-benefício: validar primeiro, melhorar depois**

---

## 📋 Plano de Ação Recomendado

### Fase 1: Treinar Outras Ações (AGORA) ⭐

**Objetivo:** Validar modelo em PETR4 e ITUB4

**Comando:**
```bash
cd ~/Arquivos/TCC/codigo/pipeline
./treinar_outros_ativos.sh
```

**Tempo estimado:** 6-8 horas (pode rodar durante a noite)

**Entregáveis:**
- Modelos treinados para PETR4 e ITUB4
- Resultados comparativos entre os 3 ativos
- Análise de robustez do modelo

**Critérios de sucesso:**
- ✅ Acurácia média > 50% em todos os ativos
- ✅ Resultados consistentes entre ativos
- ✅ Modelos salvos para todos os folds

---

### Fase 2: Análise Comparativa (Depois do Treinamento)

**Atividades:**
1. Comparar acurácias entre VALE3, PETR4, ITUB4
2. Identificar padrões (qual ativo funciona melhor?)
3. Analisar variabilidade entre folds
4. Documentar resultados para TCC

**Script de análise:**
```bash
# Criar script comparativo
uv run python src/scripts/comparar_ativos.py
```

---

### Fase 3: Melhorias do Modelo (Depois da Validação)

**Só depois de validar em múltiplos ativos, implementar:**

#### Opção A: Melhorias Técnicas (Rápido)
1. **Cosine Annealing Scheduler** (~30 min implementação)
   - Benefício esperado: +1-2% acurácia
   - Tempo treino: +1h por ativo

2. **Features Adicionais** (~1h implementação)
   - Amplitude high-low
   - Volume normalizado
   - Hora do dia (sin/cos)
   - Benefício esperado: +2-3% acurácia

#### Opção B: Ensemble (Médio Prazo)
1. **Ensemble dos 5 folds** (~1h implementação)
   - Voting ou média ponderada
   - Benefício esperado: +3-5% acurácia
   - Não precisa retreinar

2. **Metaclassificador** (Avançado)
   - Combinar CNN-LSTM + LSTM + XGBoost
   - Benefício esperado: +5-8% acurácia
   - Tempo: ~5h implementação + treino

---

## ⚠️ Sobre o Fold 5 do VALE3

**Observação:** Fold 5 teve 52.27% agora vs 56.82% anterior

**Possíveis causas:**
1. **Variabilidade normal:** Walk-forward tem variabilidade entre execuções
2. **Hiperparâmetros diferentes:** Optuna pode ter escolhido parâmetros diferentes
3. **Seed diferente:** Pode ter afetado inicialização

**Não é problema se:**
- ✅ Média geral está consistente (~53%)
- ✅ Outros folds estão OK
- ✅ Resultado ainda acima de baseline (50%)

**Ação:** Monitorar se padrão se repete em PETR4/ITUB4

---

## 📈 Expectativas Realistas

### Resultados Esperados por Ativo

| Ativo | Acurácia Esperada | Justificativa |
|-------|------------------|---------------|
| VALE3 | 52-55% | ✅ Já validado |
| PETR4 | 52-55% | Similar liquidez/volatilidade |
| ITUB4 | 50-54% | Setor financeiro pode ser diferente |

### Se Resultados Forem Consistentes:
- ✅ Modelo é robusto
- ✅ Pode prosseguir com melhorias
- ✅ TCC tem validação sólida

### Se Resultados Forem Muito Diferentes:
- ⚠️ Investigar causas (dados, features, regime de mercado)
- ⚠️ Ajustar modelo antes de melhorias
- ⚠️ Documentar limitações no TCC

---

## 🎯 Decisão Final

### ✅ RECOMENDAÇÃO: Treinar PETR4 e ITUB4 AGORA

**Razões:**
1. Validação metodológica é **prioridade** para TCC
2. Mais rápido que implementar melhorias
3. Dá base sólida para decidir próximas melhorias
4. Permite análise comparativa rica

**Próximo passo:**
```bash
./treinar_outros_ativos.sh
```

**Depois:**
- Analisar resultados
- Decidir se melhorias são necessárias
- Implementar melhorias se justificado

---

## 📝 Checklist

**Antes de treinar:**
- [ ] Verificar se GPU está disponível
- [ ] Verificar espaço em disco (modelos são ~200-300KB cada)
- [ ] Backup dos modelos atuais (opcional)

**Durante treinamento:**
- [ ] Monitorar logs periodicamente
- [ ] Verificar uso de GPU (`nvidia-smi`)
- [ ] Verificar se modelos estão sendo salvos

**Após treinamento:**
- [ ] Verificar resultados de PETR4
- [ ] Verificar resultados de ITUB4
- [ ] Comparar com VALE3
- [ ] Documentar análise comparativa
- [ ] Decidir próximos passos (melhorias ou análise)

---

**Última atualização:** 2026-01-26  
**Próximo passo:** Executar `./treinar_outros_ativos.sh`
