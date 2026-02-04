# Documentação TCC - Pipeline de Predição

Esta pasta contém toda a documentação técnica necessária para escrever o TCC.

---

## Estrutura

```
documentacao/
├── INDICE.md                     # Índice principal (comece aqui)
├── README.md                     # Este arquivo
├── ordem_cronologica.md          # Timeline de decisões e implementações
├── periodo_dados.md              # Período exato dos dados
├── decisoes_tecnicas/            # Justificativas de decisões
│   └── banda_morta.md
├── implementacoes/               # Metodologia e resultados atuais (TCC)
│   ├── baselines.md
│   ├── walk_forward_validation.md
│   ├── resultados_consolidados_2026_02_03.md
│   └── ...
├── projeto/                      # Cronograma e planejamento
│   ├── PROXIMOS_PASSOS_CONSOLIDADO.md
│   ├── CRONOGRAMA.md             # Plano 30 dias (TCC geral)
│   └── CRONOGRAMA_CODIGO.md      # Checklist desenvolvimento código
└── historico/                    # Documentação arquivada (consultar se precisar)
    ├── README.md
    ├── implementacoes/           # Correções e melhorias já incorporadas
    └── projeto/                  # Análises e guias pontuais
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

### Formato (padrão dos .md)

- **Cabeçalho:** `# Título` → linha em branco → `**Data:** YYYY-MM-DD` → `**Status:** ...` → `---` → conteúdo
- **Seções:** `## 1. Nome`, `## 2. Nome`, etc.; subseções com `###`
- **Conteúdo:** sempre em tópicos (bullets), não parágrafos longos
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
