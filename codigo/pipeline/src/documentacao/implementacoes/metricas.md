# Implementação: Métricas de Avaliação

**Data:** 2025-01-23  
**Status:** Implementado

---

## Métricas Implementadas

### Métricas Preditivas

1. **Acurácia Direcional**
   - Hit rate de previsão de direção
   - Ignora apenas valores exatamente zero
   - Base para comparação de modelos

2. **Acurácia**
   - Taxa de acertos gerais
   - Calculada sobre movimentos não-neutros

3. **Balanced Accuracy**
   - Robustez a classes desbalanceadas
   - Média de sensibilidade e especificidade

4. **F1-Score**
   - Média harmônica de precisão e recall
   - Útil quando classes estão desbalanceadas

5. **MCC (Matthews Correlation Coefficient)**
   - Correlação entre previsões e realidade
   - Varia de -1 a +1
   - Considerado mais informativo que F1 em casos desbalanceados

### Métricas Probabilísticas (quando disponível)

6. **Brier Score**
   - Qualidade das probabilidades previstas
   - Quanto menor, melhor

7. **Log-Loss**
   - Calibração das probabilidades
   - Penaliza confiança incorreta

8. **AUC-PR**
   - Área sob curva Precisão-Revocação
   - Melhor que AUC-ROC para classes desbalanceadas

---

## Características

### Sem Banda Morta
- Ignora apenas valores exatamente zero
- Usa praticamente todos os dados
- Maximiza informação disponível

### Filtragem
- Apenas movimentos não-neutros são considerados
- Neutros reais (retorno == 0) são ignorados
- Mantém foco em previsão de direção

---

## Resultados Típicos (Baselines)

| Métrica | Naive | Drift | ARIMA |
|---------|-------|-------|-------|
| Accuracy | 50.50% | 49.76% | 48.36% |
| Balanced Acc | 50.09% | 49.93% | 48.79% |
| F1-Score | 0.315 | 0.543 | 0.593 |
| MCC | 0.002 | -0.002 | -0.029 |

---

## Referências para TCC

### Seção: Metodologia - Métricas de Avaliação

**Pontos a mencionar:**
- Lista de métricas implementadas
- Justificativa de cada métrica
- Foco em métricas robustas a desbalanceamento
- Métricas probabilísticas para modelos que as fornecem

---

## Arquivos

- `src/utils/metrics.py`
