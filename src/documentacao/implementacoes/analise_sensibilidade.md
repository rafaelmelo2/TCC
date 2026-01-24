# Implementação: Análise de Sensibilidade - Walk-Forward Validation

**Data:** 2025-01-23  
**Status:** Implementado e pronto para uso

---

## Contexto

A análise de sensibilidade valida a robustez da configuração principal de walk-forward escolhida **a priori**. 

**IMPORTANTE**: Esta análise NÃO serve para escolher a melhor configuração (isso seria overfitting de configuração), mas sim para **validar** que a escolha principal é robusta.

---

## Metodologia

### Configuração Principal (Escolhida a Priori)

A configuração principal foi escolhida **antes** de executar experimentos, baseada em literatura:

- **Treino**: 1 ano (6.552 barras = 252 dias × 26 barras/dia)
- **Teste**: 1 mês (546 barras = 21 dias × 26 barras/dia)
- **Embargo**: 5 barras (~2 horas)

**Justificativa**:
- Treino de 1 ano: Permite capturar padrões sazonais e ciclos anuais (Prado, 2018)
- Teste de 1 mês: Permite avaliação frequente e detecção de mudanças de regime
- Embargo de 5 barras: Previne contaminação temporal mantendo viabilidade computacional

### Configurações de Teste

O script testa 5 configurações diferentes:

1. **mais_permissivo**: Embargo mínimo (1 barra)
2. **principal**: Configuração principal (5 barras) ⭐
3. **mais_conservador**: Embargo maior (10 barras)
4. **mais_treino**: Mais dados de treino (1.5 anos)
5. **embargo_dia**: Embargo de 1 dia completo (26 barras)

---

## Implementação

### Arquivo: `src/tests/testar_sensibilidade_walkforward.py`

**Funcionalidades**:
- Testa todas as configurações definidas em `config.py`
- Usa NaiveBaseline para validação rápida
- Calcula métricas comparativas
- Gera relatório de robustez

**Uso**:
```bash
# Teste básico
python -m src.tests.testar_sensibilidade_walkforward

# Com opções
python -m src.tests.testar_sensibilidade_walkforward --ativo VALE3 --verbose
```

### Configurações em `src/config.py`

As configurações estão definidas em `CONFIGURACOES_SENSIBILIDADE`:

```python
CONFIGURACOES_SENSIBILIDADE = [
    {
        'nome': 'mais_permissivo',
        'treino': 252 * 26,
        'teste': 21 * 26,
        'embargo': 1,
        'descricao': 'Configuração mais permissiva: embargo mínimo'
    },
    # ... outras configurações
]
```

---

## Interpretação dos Resultados

### Critérios de Robustez

- **Robusta** (✅): Variação < 2% em acurácia direcional
- **Aceitável** (⚠️): Variação < 5%
- **Não robusta** (❌): Variação >= 5%

### Exemplo de Saída

```
RESULTADOS CONSOLIDADOS
======================================================================
Configuração          | Acurácia Direcional | Diferença
----------------------------------------------------------------------
mais_permissivo       | 0.5234             | +0.0012 (+0.23%)
principal             | 0.5222             | 0.0000 (0.00%) ⭐
mais_conservador      | 0.5218             | -0.0004 (-0.08%)
mais_treino           | 0.5230             | +0.0008 (+0.15%)
embargo_dia           | 0.5215             | -0.0007 (-0.13%)

CONCLUSÃO SOBRE ROBUSTEZ
======================================================================
Diferença máxima: 0.0019 (0.36%)
✅ Configuração principal é ROBUSTA (variação < 2%)
```

---

## Quando Executar

### Momento Ideal

Execute a análise de sensibilidade **depois** de:
1. ✅ Definir configuração principal a priori
2. ✅ Implementar walk-forward validation
3. ✅ Ter baselines funcionando

Execute **antes** de:
- Finalizar resultados do TCC
- Escrever seção de metodologia
- Apresentar resultados

### Frequência

- **Uma vez**: Para validar robustez da configuração principal
- **Opcionalmente**: Se mudar configuração principal, re-executar

---

## Referências para TCC

### Seção: Metodologia - Configuração Experimental

**Pontos a mencionar**:
- Configuração principal escolhida a priori baseada em literatura
- Análise de sensibilidade realizada para validar robustez
- Resultados mostram que configuração é robusta (variação < X%)

### Seção: Resultados - Análise de Sensibilidade

**Pontos a mencionar**:
- Tabela comparativa de configurações
- Diferença máxima em relação à configuração principal
- Conclusão sobre robustez

---

## Próximos Passos

1. **Executar análise** quando tiver dados processados
2. **Documentar resultados** no TCC
3. **Incluir tabela** de comparação na seção de resultados
4. **Justificar escolha** da configuração principal

---

## Arquivos Relacionados

- `src/config.py` - Configurações definidas
- `src/tests/testar_sensibilidade_walkforward.py` - Script de teste
- `src/utils/validation.py` - WalkForwardValidator
- `src/models/baselines.py` - Modelos baseline para teste

---

## Notas Importantes

⚠️ **NÃO use esta análise para escolher configuração**: Isso seria overfitting de configuração e violaria metodologia científica.

✅ **Use para validar robustez**: Mostrar que a escolha a priori é sólida.

📊 **Documente no TCC**: Inclua tabela comparativa e conclusão sobre robustez.
