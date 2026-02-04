# Índice - Documentação TCC

**Padrão dos .md:** cabeçalho (Data, Status), seções numeradas, tópicos. Ver [README.md](README.md).

---

## Por Tópico

### Decisões Técnicas
- [Remoção da Banda Morta](decisoes_tecnicas/banda_morta.md)
- [Simplificação do Código](ordem_cronologica.md#2025-01-23---simplificação-do-código)
- [Período Exato dos Dados](periodo_dados.md)

### Implementações (metodologia e resultados atuais)
- [Arquitetura dos Modelos (LSTM, CNN-LSTM)](implementacoes/arquitetura_modelos.md)
- [Baselines](implementacoes/baselines.md)
- [Backtesting com Custos](implementacoes/backtesting.md) (TCC 4.5.1)
- [Walk-Forward Validation](implementacoes/walk_forward_validation.md)
- [Feature Engineering](implementacoes/feature_engineering.md)
- [Métricas de Avaliação](implementacoes/metricas.md)
- [Análise de Sensibilidade](implementacoes/analise_sensibilidade.md)
- [Testes Diebold-Mariano](implementacoes/testes_estatisticos_diebold_mariano.md) (TCC 4.5.2)
- [Resultados Consolidados (2026-02-03)](implementacoes/resultados_consolidados_2026_02_03.md) – checklist, interpretação DM, limitação colapso, Fase 7

### Projeto (cronograma)
- [Próximos Passos Consolidado](projeto/PROXIMOS_PASSOS_CONSOLIDADO.md) – cronograma e prioridades
- [Cronograma TCC (30 dias)](projeto/CRONOGRAMA.md) – plano geral dia a dia (22/01–20/02/2026)
- [Cronograma de Desenvolvimento – Código](projeto/CRONOGRAMA_CODIGO.md) – checklist de implementação

### Análises
- [Análise da Banda Morta](decisoes_tecnicas/banda_morta.md#análise-realizada)
- [Correção do ARIMA](ordem_cronologica.md#2025-01-23---correção-do-problema-arima)

### Histórico (arquivado)
Documentação de correções, melhorias e análises pontuais já incorporadas ao projeto. Consulta opcional.
- [Índice do histórico](historico/README.md) – lista e descrição dos arquivos em `historico/`

---

## Por Seção do TCC

### Metodologia - Engenharia de Features
- [Feature Engineering](implementacoes/feature_engineering.md)
- [Decisão sobre Banda Morta](decisoes_tecnicas/banda_morta.md)

### Metodologia - Dados
- [Período Exato dos Dados](periodo_dados.md) – 22/10/2020 até 22/10/2025 (5 anos)

### Metodologia - Validação
- [Walk-Forward Validation](implementacoes/walk_forward_validation.md)
- [Análise de Sensibilidade](implementacoes/analise_sensibilidade.md)

### Metodologia - Modelos
- [Arquitetura dos Modelos (LSTM, CNN-LSTM)](implementacoes/arquitetura_modelos.md)
- [Baselines](implementacoes/baselines.md)

### Metodologia - Métricas
- [Métricas de Avaliação](implementacoes/metricas.md)
- [Backtests e Custos de Transação](implementacoes/backtesting.md) (TCC 4.5.1)
- [Testes de Robustez e Significância (Diebold-Mariano)](implementacoes/testes_estatisticos_diebold_mariano.md) (TCC 4.5.2)

### Metodologia - Seleção de Hiperparâmetros
- Otimização com Optuna em `src/train.py` e `src/utils/optuna_optimizer.py` (early stopping, Cosine Annealing, class weights). Detalhes no [histórico](historico/README.md) (melhorias_criticas_2026_01_27.md).

### Resultados
- [Resultados Consolidados (2026-02-03)](implementacoes/resultados_consolidados_2026_02_03.md) – checklist pipeline, interpretação DM, colapso (F1/MCC), texto para TCC, Fase 7
- [Resultados dos Baselines](implementacoes/baselines.md#resultados-walk-forward-vale3)
- [Impacto da Remoção da Banda Morta](decisoes_tecnicas/banda_morta.md#impacto-mensurável)

---

## Ordem Cronológica

Ver [ordem_cronologica.md](ordem_cronologica.md) para timeline completa.

---

## Como Usar

1. **Para escrever metodologia:** Consultar `implementacoes/` (e `decisoes_tecnicas/` quando for decisão).
2. **Para resultados e próximos passos:** [resultados_consolidados_2026_02_03.md](implementacoes/resultados_consolidados_2026_02_03.md) e [PROXIMOS_PASSOS_CONSOLIDADO.md](projeto/PROXIMOS_PASSOS_CONSOLIDADO.md).
3. **Para timeline:** [ordem_cronologica.md](ordem_cronologica.md).
4. **Para decisões/correções passadas:** [historico/](historico/README.md).
