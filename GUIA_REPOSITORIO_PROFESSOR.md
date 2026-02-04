# Guia do repositório – TCC Rafael da Silva Melo

**Título do trabalho:** Predição Automática de Indicativos Financeiros para Bolsa de Valores Considerando o Aspecto Temporal  

**Objetivo:** Modelo híbrido CNN+LSTM para prever direção de movimentos de preços intradiários (barras de 15 min) em ações líquidas da B3 (PETR4, VALE3, ITUB4).

Este guia indica **onde está cada parte relevante do TCC** no repositório. Quase tudo que foi implementado está documentado em Markdown dentro do próprio código; este arquivo só orienta onde procurar. Em caso de dúvida, consulte os arquivos indicados ou entre em contato com o aluno.

---

## 1. Documentação

**Estou documentando praticamente tudo que faço.** A documentação técnica do TCC fica em:

**`codigo/pipeline/src/documentacao/`**

- **Índice (comece aqui):** `INDICE.md` – organiza tudo por tópico e por seção do TCC.
- **Implementações:** pasta `implementacoes/` – metodologia e resultados: walk-forward, feature engineering, arquitetura dos modelos, baselines, métricas, backtesting, testes Diebold-Mariano, **resultados consolidados** (interpretação dos testes e limitação do colapso de classe).
- **Decisões técnicas:** pasta `decisoes_tecnicas/` – justificativas (ex.: banda morta, período dos dados).
- **Projeto:** pasta `projeto/` – cronograma e próximos passos.
- **Histórico:** pasta `historico/` – documentação arquivada (consultar se precisar).

Os arquivos em `implementacoes/` têm relação direta com as seções do TCC (validação, dados, modelos, métricas, etc.).

---

## 2. Texto do TCC (proposta / documento)

- **PDF:** na raiz do repositório – `Predição_Automática_de_Indicativos_Financeiros_..._2025.pdf`
- **Cópia e versão em texto:** `codigo/pipeline/others/generate_text_for_ia_by_pdf/` (mesmo PDF e arquivo `.txt`)

---

## 3. Estrutura do que é parte do TCC

```
TCC/
├── GUIA_REPOSITORIO_PROFESSOR.md     ← Este guia
├── Predição_Automática_..._2025.pdf   # Texto do TCC
├── codigo/
│   ├── dados/                        # Obtenção de dados (B3, MetaTrader 5)
│   └── pipeline/                     # Pipeline principal do TCC
│       ├── data/                     # Dados: raw (OHLCV 15 min), processed, backtest, visualizações
│       ├── models/                   # Modelos treinados (checkpoints por ativo/fold)
│       └── src/
│           ├── documentacao/         # ★ Toda a documentação (INDICE.md é a entrada)
│           ├── data_processing/      # Carga, engenharia de features, sequências
│           ├── models/               # Baselines (Naive, Drift, ARIMA, Prophet) e redes (LSTM, CNN-LSTM)
│           ├── utils/                # Walk-forward, métricas, backtesting, Diebold-Mariano
│           ├── config.py             # Configurações (período dos dados, janela, banda morta)
│           └── train.py              # Treinamento com walk-forward
└── importants/                       # Materiais auxiliares (artigos, roteiro, slides) – não é código
```

---

## 4. Dados

- **Brutos (OHLCV 15 min):** `codigo/pipeline/data/raw/` – CSV por ativo. Período real: 22/10/2020 a 22/10/2025 (origem externa, ex.: MetaTrader 5). Detalhes em `src/documentacao/periodo_dados.md`.
- **Processados e resultados:** `codigo/pipeline/data/processed/` – baselines, análise dos modelos, comparativos, testes Diebold-Mariano.
- **Backtests:** `codigo/pipeline/data/backtest/` – resultados das simulações com custos.
- **Visualizações:** `codigo/pipeline/data/visualizacoes/` – gráficos de features e target por ativo/ano.

Resumo do que cada parte de `data/` contém: `codigo/pipeline/data/README.md`.

---

## 5. Código (resumo)

- **Configurações globais:** `codigo/pipeline/src/config.py` (período, colunas, janela, banda morta).
- **Fluxo:** carga e validação dos dados → engenharia de features (retornos, EMA, RSI, Bollinger, volatilidade, conforme TCC 4.2) → montagem de sequências e target (banda morta) → validação walk-forward → treino (LSTM/CNN-LSTM) e baselines → métricas e backtesting.
- **Treino:** `src/train.py`. Baselines e modelos em `src/models/`; validação e métricas em `src/utils/`.

Se for preciso **reproduzir experimentos** (rodar baselines, treino, backtests, testes DM), os comandos estão em `codigo/pipeline/COMANDOS_RODAR_TUDO.md`. O ambiente usa **uv** (instalação: `uv sync` dentro de `codigo/pipeline/`; execução: `uv run python ...`).

---

## 6. Resultados e limitações

- **Resumo e interpretação:** `codigo/pipeline/src/documentacao/implementacoes/resultados_consolidados_2026_02_03.md` – checklist do pipeline, interpretação dos testes Diebold-Mariano, limitação do colapso de classe (F1/MCC zerados), texto sugerido para o TCC.
- **Diebold-Mariano:** p-valores > 0,05 em todas as comparações CNN-LSTM vs baselines (diferença não significativa ao nível de 5%).
- **Limitação:** colapso para uma classe; metodologia (walk-forward, baselines, DM, backtest) permanece válida para o relatório.

---

## 7. Referência rápida: “Onde acho X?”

| Procuro… | Onde está |
|----------|-----------|
| Texto/PDF do TCC | Raiz: `Predição_Automática_..._2025.pdf` |
| Documentação técnica (índice) | `codigo/pipeline/src/documentacao/INDICE.md` |
| Resultados e interpretação (DM, colapso) | `src/documentacao/implementacoes/resultados_consolidados_2026_02_03.md` |
| Metodologia (walk-forward, features, modelos, métricas) | `src/documentacao/implementacoes/` e `INDICE.md` |
| Decisões (banda morta, período) | `src/documentacao/decisoes_tecnicas/` e `periodo_dados.md` |
| Cronograma e próximos passos | `src/documentacao/projeto/PROXIMOS_PASSOS_CONSOLIDADO.md` |
| Dados brutos e resultados | `codigo/pipeline/data/` (raw, processed, backtest) – ver `data/README.md` |
| Comandos para rodar experimentos | `codigo/pipeline/COMANDOS_RODAR_TUDO.md` (se precisar) |

---

**Se tiver dúvida:** use este guia e os arquivos indicados. Se não encontrar, entre em contato com o aluno (Rafael da Silva Melo).
