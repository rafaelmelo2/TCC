# Testes Estatísticos - Diebold-Mariano (TCC 4.5.2)

**Data:** 2026-02-02  
**Status:** Implementado (incl. regimes, Brier, tabelas/gráficos)  
**Referência TCC:** Seção 4.5.2 (Testes de Robustez e Significância)

## Objetivo

Comparar a série de perdas/erros do modelo proposto (CNN-LSTM) contra os baselines por meio do teste de Diebold-Mariano (DIEBOLD; MARIANO, 1995), avaliando a significância estatística das diferenças de acurácia direcional e de Brier. Segmentar por regimes de volatilidade (calmaria vs. choques) para verificar estabilidade.

## Componentes

### Módulo src/utils/diebold_mariano.py

- perda_direcional(y_true, y_pred): Perda 0/1 por observação (não neutros).
- perda_brier(y_true, y_prob): Perda Brier (y_prob - y_bin)^2 para não neutros.
- diebold_mariano(loss_a, loss_b, h=1): Teste DM; H0: mesma acurácia; retorna estatística e p-valor.
- resumo_dm(...): Dicionário com n_obs, perdas médias, DM_statistic, DM_pvalue.
- segmentar_por_volatilidade(volatility, loss_a, loss_b, percentil=50): Segmenta perdas em baixa/alta volatilidade (mediana).
- teste_paired_folds(acc_a, acc_b): Teste pareado por folds (Wilcoxon ou t-test).

### Script src/scripts/rodar_testes_estatisticos.py

Gera folds walk-forward, obtém previsões CNN-LSTM e de cada baseline, alinha no mesmo conjunto de barras (trim_baseline_pred_to_cnn), calcula perdas direcionais e aplica o teste DM. Saída: data/processed/testes_diebold_mariano.csv (coluna Regime: geral, baixa_vol, alta_vol, brier).

Uso:
- uv run python src/scripts/rodar_testes_estatisticos.py --ativo PETR4
- uv run python src/scripts/rodar_testes_estatisticos.py --todos
- uv run python src/scripts/rodar_testes_estatisticos.py --todos --regimes   (DM por regime de volatilidade)
- uv run python src/scripts/rodar_testes_estatisticos.py --todos --brier     (DM sobre perda Brier)

Interpretação: DM_pvalue menor que 0,05 indica diferença significativa entre CNN-LSTM e o baseline. Diferenca_perda menor que zero indica CNN melhor.

### Script src/scripts/gerar_tabelas_graficos_dm.py

Lê o CSV de resultados DM e gera tabelas resumo (p-valores com estrelas de significância, diferença de perda) e opcionalmente heatmap de p-valores.

Uso:
- uv run python src/scripts/gerar_tabelas_graficos_dm.py
- uv run python src/scripts/gerar_tabelas_graficos_dm.py --csv data/processed/testes_diebold_mariano.csv --saida_dir data/processed --grafico

Saídas: dm_resumo_pvalores.csv, dm_diferenca_perda_geral.csv, dm_heatmap_pvalores.png (se --grafico).

## Alinhamento

CNN usa janela 60 barras; baselines são cortados para o mesmo conjunto de barras por fold (trim_baseline_pred_to_cnn).

## Regimes de volatilidade

Usa a coluna volatility de df_features (janela 20). Em cada comparação, observações não neutras são divididas em baixa_vol (volatilidade <= mediana) e alta_vol (volatilidade > mediana). O teste DM é aplicado em cada segmento (mín. 10 obs).

## Arquivos

- src/utils/diebold_mariano.py
- src/scripts/rodar_testes_estatisticos.py
- src/scripts/gerar_tabelas_graficos_dm.py
- Saída: data/processed/testes_diebold_mariano.csv
- Tabelas/figuras: data/processed/dm_resumo_pvalores.csv, dm_diferenca_perda_geral.csv, dm_heatmap_pvalores.png

---

## Interpretação dos resultados (2026-02-03)

Resultados gerados com os dados atuais (regime geral, perda direcional):

- **Heatmap (`dm_heatmap_pvalores.png`):** Todos os p-valores ficam **acima de 0,05** para os três ativos (ITUB4, PETR4, VALE3) vs os quatro baselines (ARIMA, Drift, Naive, Prophet). Ou seja: **não há diferença estatisticamente significativa** entre o CNN-LSTM e os baselines na perda de acurácia direcional.
- **Tabela de diferença de perda:** Valores negativos indicam CNN-LSTM com perda menor (melhor), mas como os p-valores são altos, essa vantagem **não é estatisticamente significativa**. O resultado é válido para reportar no TCC como “não significativo”.

Documentação completa da interpretação, checklist de conclusão das etapas 1–5 e limitações (colapso dos modelos) está em **[Resultados consolidados (2026-02-03)](resultados_consolidados_2026_02_03.md)**.