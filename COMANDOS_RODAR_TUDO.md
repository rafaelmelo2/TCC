# Comandos para rodar tudo, salvar e passar para a próxima fase

Execute a partir do diretório do pipeline: `cd /home/rafael/Arquivos/TCC/codigo/pipeline`

**Resumo completo para reunião com professor e busca:** [RESUMO_TCC_REUNIAO_PROFESSOR.md](RESUMO_TCC_REUNIAO_PROFESSOR.md) (tudo que foi feito + links para cada arquivo + perguntas que o professor pode fazer).

---

## 1. Baselines (walk-forward) – 3 ativos

Gera: `data/processed/{ATIVO}_baselines_walkforward.csv`

```bash
uv run python src/tests/testar_baselines_walkforward.py --todos
```

*(Ou por ativo: `--ativo PETR4`, `--ativo VALE3`, `--ativo ITUB4`)*

---

## 2. Análise dos modelos CNN-LSTM salvos – 3 ativos

Gera: `data/processed/{ATIVO}_cnn_lstm_analise_modelos.csv`

```bash
uv run python src/scripts/analisar_modelos_salvos.py --ativo PETR4
uv run python src/scripts/analisar_modelos_salvos.py --ativo VALE3
uv run python src/scripts/analisar_modelos_salvos.py --ativo ITUB4
```

---

## 3. Tabela comparativa CNN-LSTM vs baselines

Gera: `data/processed/comparativo_cnn_lstm_vs_baselines.csv`

```bash
uv run python src/scripts/comparar_modelos.py
```

---

## 4. Testes estatísticos (Diebold-Mariano) – CNN-LSTM vs baselines

Gera: `data/processed/testes_diebold_mariano.csv`

```bash
# Apenas acurácia direcional
uv run python src/scripts/rodar_testes_estatisticos.py --todos

# Com segmentação por regime de volatilidade (baixa/alta)
uv run python src/scripts/rodar_testes_estatisticos.py --todos --regimes

# Com teste DM sobre perda Brier
uv run python src/scripts/rodar_testes_estatisticos.py --todos --brier
```

**Gerar tabelas e gráficos a partir do CSV:**

```bash
uv run python src/scripts/gerar_tabelas_graficos_dm.py
uv run python src/scripts/gerar_tabelas_graficos_dm.py --grafico
```

Gera: `data/processed/dm_resumo_pvalores.csv`, `dm_diferenca_perda_geral.csv`, opcionalmente `dm_heatmap_pvalores.png`.

---

## 5. Backtests com custos – todos os ativos, folds e estratégias

Gera: `data/backtest/{ATIVO}_fold{FOLD}_{ESTRATEGIA}_{AAAAMMDD}_{HHMMSS}.csv` e `historico_backtest.csv`

### Opção A: Rodar um por um (controle total)

**Long/short (3 ativos × 5 folds = 15 runs):**

```bash
for ativo in PETR4 VALE3 ITUB4; do
  for fold in 1 2 3 4 5; do
    uv run python src/scripts/rodar_backtest.py --ativo $ativo --fold $fold --estrategia long_short
  done
done
```

**Long-only (3 ativos × 5 folds = 15 runs):**

```bash
for ativo in PETR4 VALE3 ITUB4; do
  for fold in 1 2 3 4 5; do
    uv run python src/scripts/rodar_backtest.py --ativo $ativo --fold $fold --estrategia long_only
  done
done
```

**Com sensibilidade a custos (um exemplo por ativo):**

```bash
uv run python src/scripts/rodar_backtest.py --ativo VALE3 --fold 1 --estrategia long_short --sensibilidade
uv run python src/scripts/rodar_backtest.py --ativo PETR4 --fold 1 --estrategia long_short --sensibilidade
uv run python src/scripts/rodar_backtest.py --ativo ITUB4 --fold 1 --estrategia long_short --sensibilidade
```

### Opção B: Usar o script em lote (todos os backtests)

```bash
./scripts/rodar_todos_backtests.sh
```

*(Script criado abaixo; roda 3 ativos × 5 folds × 2 estratégias = 30 backtests.)*

---

## 6. Resumo: ordem sugerida para “rodar tudo e salvar”

```bash
cd /home/rafael/Arquivos/TCC/codigo/pipeline

# 1) Baselines
uv run python src/tests/testar_baselines_walkforward.py --todos

# 2) Análise CNN-LSTM
uv run python src/scripts/analisar_modelos_salvos.py --ativo PETR4
uv run python src/scripts/analisar_modelos_salvos.py --ativo VALE3
uv run python src/scripts/analisar_modelos_salvos.py --ativo ITUB4

# 3) Comparativo
uv run python src/scripts/comparar_modelos.py

# 4) Testes Diebold-Mariano (opcional: --regimes, --brier)
# uv run python src/scripts/rodar_testes_estatisticos.py --todos
# uv run python src/scripts/gerar_tabelas_graficos_dm.py --grafico

# 5) Backtests (long_short + long_only para os 3 ativos, 5 folds)
for ativo in PETR4 VALE3 ITUB4; do
  for fold in 1 2 3 4 5; do
    uv run python src/scripts/rodar_backtest.py --ativo $ativo --fold $fold --estrategia long_short
    uv run python src/scripts/rodar_backtest.py --ativo $ativo --fold $fold --estrategia long_only
  done
done
```

Tudo fica salvo em:

- `data/processed/` – baselines, análise CNN-LSTM, comparativo, testes DM (testes_diebold_mariano.csv, dm_*)
- `data/backtest/` – um CSV por backtest + `historico_backtest.csv`

---

## 7. Próxima fase (após rodar e salvar)

### Checklist: o que já está salvo (verificação 2026-02-03 – concluído)

| Etapa | Esperado | Status |
|-------|----------|--------|
| 1. Baselines | `{PETR4,VALE3,ITUB4}_baselines_walkforward.csv` | ✅ Todos presentes |
| 2. Análise CNN-LSTM | `{ATIVO}_cnn_lstm_analise_modelos.csv` | ✅ PETR4, VALE3, ITUB4 |
| 3. Comparativo | `comparativo_cnn_lstm_vs_baselines.csv` | ✅ Presente |
| 4. Testes DM | `testes_diebold_mariano.csv` | ✅ Presente |
| 4b. Tabelas/gráficos DM | `dm_resumo_pvalores.csv`, `dm_diferenca_perda_geral.csv`, `dm_heatmap_pvalores.png` | ✅ Gerados |
| 5. Backtests | 3 ativos × 5 folds × 2 estratégias + `historico_backtest.csv` | ✅ Completos (30 runs) |

**Conclusão:** Todas as etapas 1–5 estão concluídas. **Fase 7 em andamento.** Interpretação dos resultados DM e limitações (colapso dos modelos) documentadas em `src/documentacao/implementacoes/resultados_consolidados_2026_02_03.md`.

---

### Itens da fase 7 (conforme [PROXIMOS_PASSOS_CONSOLIDADO.md](src/documentacao/projeto/PROXIMOS_PASSOS_CONSOLIDADO.md))

1. **Testes estatísticos (Diebold-Mariano)** – já implementado: `rodar_testes_estatisticos.py` (--regimes, --brier) e `gerar_tabelas_graficos_dm.py`.
2. **Análise de sensibilidade** – `testar_sensibilidade_walkforward.py` por ativo (e possíveis extensões).
3. **Visualizações para o TCC** – gráficos de performance, curvas de calibração, comparação com baselines.
4. **Consolidação de resultados** – tabelas finais e texto para o relatório.
