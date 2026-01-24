# Documentação TCC - Pipeline de Predição

Esta pasta contém toda a documentação técnica necessária para escrever o TCC.

---

## Estrutura

```
documentacao/
├── ordem_cronologica.md          # Timeline completa de decisões e implementações
├── decisoes_tecnicas/            # Justificativas detalhadas de decisões
│   ├── banda_morta.md
│   └── ...
└── implementacoes/               # Documentação de funcionalidades
    ├── baselines.md
    ├── walk_forward_validation.md
    └── ...
```

---

## Como Usar

### Para Escrever o TCC

1. **Consultar `ordem_cronologica.md`**
   - Ver timeline completa
   - Entender sequência de decisões
   - Identificar datas e contexto

2. **Consultar `decisoes_tecnicas/`**
   - Justificativas detalhadas
   - Análises realizadas
   - Impacto mensurável

3. **Consultar `implementacoes/`**
   - Detalhes técnicos
   - Resultados obtidos
   - Referências para seções do TCC

### Formato

- **Sempre em tópicos** (não parágrafos)
- **Dados concretos** (métricas, percentuais)
- **Justificativas claras** (por quê)
- **Impacto mensurável** (antes/depois)

---

## Manutenção

Esta documentação é mantida automaticamente pelo subagent `documentador-tcc`.

Para adicionar manualmente:
1. Criar arquivo em `decisoes_tecnicas/` ou `implementacoes/`
2. Atualizar `ordem_cronologica.md`
3. Seguir formato de tópicos

---

## Referências para LaTeX

Cada arquivo contém seção "Referências para TCC" indicando:
- Onde mencionar no TCC
- Quais pontos destacar
- Quais métricas incluir
