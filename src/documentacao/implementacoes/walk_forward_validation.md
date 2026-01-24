# Implementação: Walk-Forward Validation

**Data:** 2025-01-23  
**Status:** Implementado e testado

---

## Contexto

- Validação walk-forward obrigatória para séries temporais
- K-fold tradicional viola ordem temporal
- Necessário para evitar data leakage

---

## Implementação

### Classe: WalkForwardValidator

**Características:**
- Divisão sequencial de dados
- Suporte a embargo temporal
- Agregação de resultados por fold
- Compatível com qualquer modelo

### Parâmetros Padrão
- Treino: 6,552 barras (~1 ano: 252 dias × 26 barras/dia)
- Teste: 546 barras (~1 mês: 21 dias × 26 barras/dia)
- Embargo: 1 barra

### Estrutura de Folds

```
Fold 1: [Treino: 0-6552] [Embargo: 6552-6553] [Teste: 6553-7099]
Fold 2: [Treino: 7099-13651] [Embargo: 13651-13652] [Teste: 13652-14198]
...
```

### Funcionalidades
- Geração automática de folds
- Validação com funções customizadas (fit_func, predict_func)
- Agregação de métricas por fold
- Resultados globais concatenados

---

## Justificativa Metodológica

### Por que walk-forward?
- Respeita ordem temporal dos dados
- Simula uso real do modelo (treinar no passado, prever futuro)
- Evita data leakage
- Permite avaliação de performance ao longo do tempo

### Por que embargo?
- Previne contaminação entre treino e teste
- Simula delay real de informações
- Padrão em validação temporal

---

## Resultados de Teste

### Baselines Testados
- 5 folds gerados para VALE3
- 2,388 amostras de teste no total
- Todos os baselines executados com sucesso

### Métricas Agregadas
- Por fold: métricas individuais
- Global: todas as previsões concatenadas
- Estatísticas: média, desvio-padrão, min, max

---

## Referências para TCC

### Seção: Metodologia - Validação

**Pontos a mencionar:**
- Walk-forward validation implementada
- Configuração: 1 ano treino, 1 mês teste
- Embargo de 1 barra
- Justificativa: evitar data leakage em séries temporais

### Seção: Resultados - Protocolo de Validação

**Pontos a mencionar:**
- Número de folds gerados
- Tamanho dos conjuntos de treino/teste
- Agregação de resultados

---

## Arquivos

- `src/utils/validation.py` - Implementação
- `src/tests/testar_baselines_walkforward.py` - Testes
