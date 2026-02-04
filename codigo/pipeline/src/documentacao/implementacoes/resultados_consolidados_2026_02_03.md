# Resultados consolidados e interpretação – Pronto para Fase 7

**Data:** 2026-02-03  
**Status:** Etapas 1–5 do pipeline concluídas; **pronto para Fase 7** (sensibilidade, visualizações, consolidação para o relatório).  
**Referência:** COMANDOS_RODAR_TUDO.md (seções 1–7)

---

## 1. Checklist de conclusão (etapas 1–5)

| Etapa | Esperado | Status |
|-------|----------|--------|
| 1. Baselines | `{PETR4,VALE3,ITUB4}_baselines_walkforward.csv` | ✅ Todos presentes |
| 2. Análise CNN-LSTM | `{ATIVO}_cnn_lstm_analise_modelos.csv` | ✅ PETR4, VALE3, ITUB4 |
| 3. Comparativo | `comparativo_cnn_lstm_vs_baselines.csv` | ✅ Presente |
| 4. Testes DM | `testes_diebold_mariano.csv` | ✅ Presente |
| 4b. Tabelas/gráficos DM | `dm_resumo_pvalores.csv`, `dm_diferenca_perda_geral.csv`, `dm_heatmap_pvalores.png` | ✅ Gerados |
| 5. Backtests | 3 ativos × 5 folds × 2 estratégias + `historico_backtest.csv` | ✅ Completos (30 runs) |

**Conclusão:** Todas as saídas necessárias estão em `data/processed/` e `data/backtest/`. Pode-se iniciar a **Fase 7** (análise de sensibilidade, visualizações para o TCC, consolidação de resultados).

---

## 2. Interpretação dos resultados Diebold-Mariano

### 2.1 Heatmap de p-valores (`dm_heatmap_pvalores.png`)

- **Título:** "Teste Diebold-Mariano: p-valor (CNN-LSTM vs baseline)". Legenda: "< 0,05 = diferença significativa".
- **Eixos:** Ativos (ITUB4, PETR4, VALE3) × Baselines (ARIMA, Drift, Naive, Prophet).
- **Resultado observado:** Todas as células apresentam p-valores **acima de 0,05** (visualmente na faixa vermelha da escala).
- **Interpretação:** Não há **diferença estatisticamente significativa** entre o CNN-LSTM e nenhum dos baselines (ARIMA, Drift, Naive, Prophet) para os três ativos, no regime geral, ao nível de 5%. Ou seja: não rejeitamos a hipótese nula de igualdade de desempenho (em termos da perda de acurácia direcional).

**Para o TCC:** O teste de Diebold-Mariano foi aplicado conforme metodologia (Seção 4.5.2). O resultado é válido e deve ser reportado como “não significativo” — o modelo proposto não supera os baselines de forma estatisticamente significativa na métrica de perda direcional.

### 2.2 Tabela de p-valores (`dm_resumo_pvalores.csv`)

Valores típicos (regime geral):

- **ITUB4:** ~0,25 (ARIMA, Drift, Naive); ~0,59 (Prophet).
- **PETR4:** ~0,35 a ~0,77.
- **VALE3:** ~0,32 a ~0,92.

Nenhum p-valor < 0,05; portanto nenhuma estrela de significância (*, **, ***) é atribuída. Consistente com o heatmap.

### 2.3 Tabela de diferença de perda (`dm_diferenca_perda_geral.csv`)

- **Convenção:** Diferença = perda CNN-LSTM − perda baseline. **Negativo** = CNN-LSTM melhor (menor perda).
- **Exemplos:** ITUB4 com diferenças negativas vs todos os baselines (CNN ligeiramente melhor em perda); PETR4 e VALE3 com mistura de valores negativos e positivos.
- **Interpretação:** Mesmo quando a diferença de perda é negativa (CNN melhor), o teste DM indica que essa diferença **não é estatisticamente significativa** (p > 0,05). Portanto: possível vantagem pontual em perda, mas não sustentada estatisticamente.

**Para o TCC:** Pode-se reportar que “em alguns ativos e baselines a perda média do CNN-LSTM foi ligeiramente menor, porém o teste de Diebold-Mariano não indicou significância estatística em nenhuma comparação”.

---

## 3. Limitação: colapso dos modelos (F1 e MCC zerados)

### 3.1 O que foi observado

Na análise dos modelos salvos (`analisar_modelos_salvos.py`) para PETR4, VALE3 e ITUB4:

- Em **todos os folds** e nos três ativos, as previsões de **classe** são praticamente só uma: **Alta=0, Baixa=N** (modelo prediz sempre “Baixa”).
- **F1-Score** e **MCC** ficam **0,0** em todos os folds.
- **Acurácia direcional** permanece em torno de 50% porque, ao prever sempre “Baixa”, o modelo acerta quando o real é “Baixa” e erra quando é “Alta” ou “Neutro”; com distribuição equilibrada, a acurácia fica próxima do acaso.

Ou seja: o modelo **colapsou para uma única classe** e não está discriminando as duas classes de forma útil.

### 3.2 Implicações

- A **metodologia e o pipeline** (walk-forward, baselines, DM, backtest) estão corretos e podem ser descritos e defendidos no TCC.
- Os **resultados de acurácia direcional** (~50%) refletem esse colapso e não uma discriminação real entre alta/baixa.
- **F1 e MCC** não devem ser usados como evidência de qualidade discriminativa (ficam zerados); a acurácia sozinha é enganosa nesse cenário.

### 3.3 Como reportar no TCC

Sugestão de redação para a seção de resultados/limitações:

- “Os modelos CNN-LSTM treinados tenderam a prever predominantemente uma única classe (colapso de classe), resultando em F1-Score e MCC nulos e acurácia direcional próxima ao acaso. Isso indica limitação na capacidade discriminativa dos modelos atuais, sem invalidar a metodologia de validação walk-forward, a comparação com baselines ou os testes de significância (Diebold-Mariano) realizados.”
- “Trabalhos futuros podem explorar balanceamento de classes, funções de perda alternativas (ex.: focal loss já utilizada) e monitoramento da distribuição de previsões durante o treino para mitigar o colapso.”

---

## 4. Fase 7 – Próximos passos

Conforme COMANDOS_RODAR_TUDO.md e PROXIMOS_PASSOS_CONSOLIDADO.md:

1. **Testes estatísticos (Diebold-Mariano)** – Concluído (incl. tabelas e heatmap).
2. **Análise de sensibilidade** – Executar `testar_sensibilidade_walkforward.py` por ativo e extensões se desejado.
3. **Visualizações para o TCC** – Gráficos de performance por fold, curvas de calibração, comparação com baselines.
4. **Consolidação de resultados** – Tabelas finais e texto para o relatório (capítulo de resultados e conclusões).

---

## 5. Arquivos de referência

| Conteúdo | Arquivo |
|----------|---------|
| Comandos do pipeline | `codigo/pipeline/COMANDOS_RODAR_TUDO.md` |
| Implementação DM | `src/documentacao/implementacoes/testes_estatisticos_diebold_mariano.md` |
| Resultados DM (tabelas/figura) | `data/processed/dm_resumo_pvalores.csv`, `dm_diferenca_perda_geral.csv`, `dm_heatmap_pvalores.png` |
| Histórico de backtests | `data/backtest/historico_backtest.csv` |
| Próximos passos gerais | `codigo/pipeline/PROXIMOS_PASSOS_CONSOLIDADO.md` |

---

**Última atualização:** 2026-02-03
