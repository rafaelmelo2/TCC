# Implementação: Capítulos 5 e 6 do TCC com dados reais

**Data:** 2026-02-16  
**Status:** Concluído  
**Arquivos:** `TCC UFG/tex/05-resultados-discussao.tex`, `06-consideracoes-finais.tex`, `images/dm_heatmap_pvalores.png`

---

## Contexto

- Capítulo 5 (Resultados e Discussão) e Capítulo 6 (Considerações Finais) precisavam ser preenchidos com dados reais do pipeline
- Dados disponíveis em CSVs: `comparativo_cnn_lstm_vs_baselines.csv`, `*_cnn_lstm_analise_modelos.csv`, `testes_diebold_mariano.csv`, `historico_backtest.csv`, `dm_resumo_pvalores.csv`
- Script `gerar_tabelas_graficos_dm.py` gerou/atualizou heatmap DM
- Este é o capítulo mais importante do PFC2

---

## O que foi feito

### Capítulo 5 – Reescrita completa (~632 linhas)
- Preenchido inteiramente com dados reais extraídos dos CSVs do pipeline
- Organizado em 9 seções conforme estrutura abaixo

### Capítulo 6 – Reescrita completa
- Refletir resultados reais (não genéricos)
- Contribuições, resultados principais, discussão sobre eficiência de mercado, trabalhos futuros

### Arquivos modificados
- `TCC UFG/tex/05-resultados-discussao.tex` (reescrito inteiro)
- `TCC UFG/tex/06-consideracoes-finais.tex` (reescrito inteiro)
- `images/dm_heatmap_pvalores.png` (gerado/atualizado via `gerar_tabelas_graficos_dm.py`)

---

## Decisões técnicas

### 1. Estrutura do Capítulo 5 (9 seções)
- Visão Geral dos resultados e organização dos artefatos
- Comparação Geral (CNN-LSTM vs baselines)
- Estabilidade por Fold
- Calibração de Probabilidades (proba_mean/proba_std)
- Teste Diebold-Mariano
- Backtests
- Regimes de Volatilidade
- Limitações e Colapso de Classe
- Síntese

### 2. Substituição: Optuna/top-10 → Calibração de Probabilidades
- Não havia CSV com configurações Optuna
- Seção de Optuna/top-10 removida
- Adicionada seção de Calibração de Probabilidades (proba_mean, proba_std)

### 3. Tabelas ajustadas para 4 folds
- Template original previa 5 folds
- Ajustado para 4 folds conforme volume efetivo dos dados

### 4. Colunas removidas/substituídas na tabela comparativa
- Bal.Acc e Brier removidas
- Motivo: dados não disponíveis para CNN-LSTM
- F1-Score utilizado como substituto

### 5. Abordagem honesta sobre resultados
- Colapso de classe identificado (MCC=0 em 10/12 folds)
- DM sem significância estatística reportado
- Backtests mistos documentados
- Seção dedicada a Limitações

### 6. Seção de Limitações (5.8)
- Colapso de classe descrito
- Possíveis causas discutidas
- O que não é invalidado (metodologia, walk-forward, DM, baselines)

---

## Dados chave reportados

### CNN-LSTM
- Hit rate: 50,76% a 51,55% (marginal sobre 50%)

### Baselines
- Hit rate: 48,46% a 51,21%
- Faixa similar entre modelos

### Teste Diebold-Mariano
- **Perda direcional**: nenhum p < 0,05 → sem significância estatística
- **Perda Brier**: todos p < 0,001 → CNN-LSTM melhor por natureza probabilística

### Backtests
- PETR4 long/short: +4,10% (concentrado em fold 1)
- VALE3 e ITUB4: ~0% (resultados neutros)

---

## Justificativa

- Este é o capítulo mais importante do PFC2
- Abordagem honesta sobre resultados neutros/negativos é contribuição válida
- Mostra que deep learning não supera baselines simples em dados intradiários de 15 min
- Validação rigorosa (walk-forward, DM, embargo) foi mantida
- Documentação transparente fortalece o trabalho acadêmico

---

## Referências

| Conteúdo        | Arquivo                                         |
|-----------------|--------------------------------------------------|
| Capítulo 5      | `TCC UFG/tex/05-resultados-discussao.tex`        |
| Capítulo 6      | `TCC UFG/tex/06-consideracoes-finais.tex`       |
| Heatmap DM      | `images/dm_heatmap_pvalores.png`                 |
| CSVs fonte      | `data/processed/`, `data/backtest/`              |
| Resultados DM   | `src/documentacao/implementacoes/resultados_consolidados_2026_02_03.md` |
