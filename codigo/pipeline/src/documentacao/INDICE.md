# Índice - Documentação TCC

Guia rápido para encontrar informações específicas.

---

## Por Tópico

### Decisões Técnicas
- [Remoção da Banda Morta](decisoes_tecnicas/banda_morta.md)
- [Simplificação do Código](ordem_cronologica.md#2025-01-23---simplificação-do-código)
- [Período Exato dos Dados](periodo_dados.md)

### Implementações
- [Baselines](implementacoes/baselines.md)
- [Walk-Forward Validation](implementacoes/walk_forward_validation.md)
- [Feature Engineering](implementacoes/feature_engineering.md)
- [Métricas de Avaliação](implementacoes/metricas.md)
- [Análise de Sensibilidade](implementacoes/analise_sensibilidade.md)
- [Correções do Treinamento (2026-01-23)](implementacoes/correcoes_treinamento_2026_01_23.md)
- [Melhorias Técnicas (2026-01-23)](implementacoes/melhorias_tecnicas_2026_01_23.md) - Gradient clipping, salvamento de modelos
- [Melhorias Críticas (2026-01-27)](implementacoes/melhorias_criticas_2026_01_27.md) - Cosine Annealing Scheduler, Class Weights melhorados
- [Mudanças Completas (2026-01-23/24)](implementacoes/mudancas_completas_2026_01_23_24.md) - Documentação completa de todas as implementações

### Análises
- [Análise da Banda Morta](decisoes_tecnicas/banda_morta.md#análise-realizada)
- [Correção do ARIMA](ordem_cronologica.md#2025-01-23---correção-do-problema-arima)
- [Correções Críticas no Treinamento](implementacoes/correcoes_treinamento_2026_01_23.md)

---

## Por Seção do TCC

### Metodologia - Engenharia de Features
- [Feature Engineering](implementacoes/feature_engineering.md)
- [Decisão sobre Banda Morta](decisoes_tecnicas/banda_morta.md)
- [Aplicação Correta da Banda Morta](implementacoes/correcoes_treinamento_2026_01_23.md#problema-1-banda-morta-não-aplicada-bug-crítico)

### Metodologia - Dados
- [Período Exato dos Dados](periodo_dados.md) - 22/10/2020 até 22/10/2025 (5 anos)

### Metodologia - Validação
- [Walk-Forward Validation](implementacoes/walk_forward_validation.md)
- [Análise de Sensibilidade](implementacoes/analise_sensibilidade.md)

### Metodologia - Modelos Baseline
- [Baselines](implementacoes/baselines.md)

### Metodologia - Métricas
- [Métricas de Avaliação](implementacoes/metricas.md)

### Metodologia - Seleção de Hiperparâmetros
- [Otimização com Optuna](implementacoes/correcoes_treinamento_2026_01_23.md#problema-3-convergência-insuficiente)
  - Early stopping com patience=10
  - Máximo de 100 épocas
  - Ajustes de learning rate
- [Melhorias Críticas (2026-01-27)](implementacoes/melhorias_criticas_2026_01_27.md)
  - Cosine Annealing Scheduler (TCC Seção 4.4)
  - Class Weights melhorados (sklearn)
  - Monitoramento de distribuição de previsões

### Resultados
- [Resultados dos Baselines](implementacoes/baselines.md#resultados-walk-forward-vale3)
  - Análise completa: Naive, Drift, ARIMA, Prophet
  - Interpretação: todos próximos de 50% (esperado)
  - Baseline estabelecido para comparação com deep learning
- [Impacto da Remoção da Banda Morta](decisoes_tecnicas/banda_morta.md#impacto-mensurável)
- [Resultados do Treinamento CNN-LSTM](implementacoes/correcoes_treinamento_2026_01_23.md#resultados-observados)
  - Correções aplicadas e impacto
  - Acurácia: ~53% (acima de baseline)
  - Análise de problemas e limitações
- [Melhorias Críticas (2026-01-27)](implementacoes/melhorias_criticas_2026_01_27.md)
  - Correção de F1=0.0 e MCC=0.0
  - Implementação de técnicas do TCC faltantes
  - Resultados esperados após melhorias

---

## Ordem Cronológica

Ver [ordem_cronologica.md](ordem_cronologica.md) para timeline completa.

---

## Como Usar

1. **Para escrever metodologia:** Consultar `implementacoes/`
2. **Para justificar decisões:** Consultar `decisoes_tecnicas/`
3. **Para timeline:** Consultar `ordem_cronologica.md`
4. **Para dados específicos:** Usar busca nos arquivos
