# Predição Automática de Indicativos Financeiros para Bolsa de Valores (TCC)

Repositório do Trabalho de Conclusão de Curso (UFG) sobre predição da direção de movimentos de preço intradiários (barras de 15 minutos) para ações líquidas da B3, com foco em rigor temporal e reprodutibilidade.

O núcleo técnico está em `codigo/pipeline/`, onde ficam o treinamento dos modelos, os baselines, os testes estatísticos, os backtests e toda a documentação de implementação.

## Visão Geral

Este trabalho investiga um pipeline de modelagem temporal para mercado financeiro, comparando um modelo híbrido CNN-LSTM com baselines clássicos.

A proposta segue validação walk-forward com embargo temporal para reduzir vazamento de informação e aproximar o cenário real de uso. Ao longo do pipeline, são gerados artefatos de desempenho estatístico e financeiro, além de documentação das decisões técnicas para escrita do TCC.

## Objetivo do Projeto

Construir, avaliar e documentar um sistema reprodutível de predição direcional para ativos da B3 (PETR4, VALE3 e ITUB4), cobrindo:

- preparação e engenharia de atributos de séries temporais intradiárias;
- treinamento e comparação de modelos baselines e deep learning;
- validação temporal robusta (walk-forward);
- avaliação estatística e de trading (incluindo custos de transação);
- consolidação de resultados para relatório acadêmico.

## Estrutura do Repositório

```text
TCC/
├── codigo/pipeline/          # Código principal, dados, scripts e docs técnicas
├── TCC UFG/                  # Texto do TCC (LaTeX)
├── artigo-revista/           # Artigo em LaTeX
├── importCommands.md         # Anotações/comandos auxiliares
└── README.md                 # Este arquivo
```

Dentro de `codigo/pipeline/`:

- `src/`: implementação de modelos, validação, métricas, scripts e testes;
- `data/`: diretórios de dados brutos, processados e backtests;
- `scripts/`: scripts shell auxiliares;
- `src/documentacao/`: documentação técnica e cronograma;
- `COMANDOS_RODAR_TUDO.md`: sequência completa de execução do pipeline;
- `rodar_pipeline_completo.sh`: execução automatizada ponta a ponta.

## Stack e Ambiente

- **Sistema recomendado:** WSL/Linux
- **Linguagem:** Python
- **Gerenciador de ambiente/dependências:** `uv`
- **Modelagem:** TensorFlow/Keras, scikit-learn, statsmodels, Prophet

## Branch para Reprodução no GitHub

Para facilitar a reprodução por qualquer pessoa, este projeto mantém uma branch dedicada ao pipeline:

- **Branch de reprodução:** `pipeline-only`

Essa branch concentra o fluxo de execução do pipeline de forma direta para quem quiser replicar os resultados do modelo e do trabalho, sem depender da organização completa do restante do repositório.

## Como Rodar (Passo a Passo)

### 1) Clonar o projeto

```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd TCC
git checkout pipeline-only
```

### 2) Entrar no diretório do pipeline

```bash
cd codigo/pipeline
```

### 3) Instalar dependências com `uv`

```bash
uv sync
```

### 4) Preparar os dados brutos

Coloque os CSVs OHLCV em `codigo/pipeline/data/raw/` com nomes no padrão usado pelo projeto (exemplo: `PETR4_M15_20201022_20251022.csv`).

As colunas esperadas estão descritas em `codigo/pipeline/README.md`.

### 5) Executar o pipeline completo

```bash
chmod +x rodar_pipeline_completo.sh
./rodar_pipeline_completo.sh
```

Modos úteis:

```bash
./rodar_pipeline_completo.sh --sem-gpu
./rodar_pipeline_completo.sh --rapido
```

Se preferir rodar por etapas (baselines -> análise -> comparativo -> DM -> backtests), siga:

`codigo/pipeline/COMANDOS_RODAR_TUDO.md`

## Execução por Etapas (atalhos)

Ainda dentro de `codigo/pipeline/`:

```bash
# Baselines (walk-forward) para todos os ativos
uv run python src/tests/testar_baselines_walkforward.py --todos

# Comparativo entre modelos
uv run python src/scripts/comparar_modelos.py

# Testes estatísticos (Diebold-Mariano)
uv run python src/scripts/rodar_testes_estatisticos.py --todos --regimes --brier
```

## Metodologia Resumida

- validação walk-forward com embargo temporal;
- comparação com baselines obrigatórios (Naive, Drift, ARIMA e Prophet);
- treinamento do modelo híbrido CNN-LSTM;
- análise por múltiplas métricas de classificação e desempenho financeiro;
- testes estatísticos para comparação de desempenho entre modelos.

Os detalhes estão documentados em `codigo/pipeline/src/documentacao/`.

## Principais Saídas Geradas

Após a execução do pipeline, os resultados ficam principalmente em:

- `codigo/pipeline/data/processed/`: métricas, comparativos e testes estatísticos;
- `codigo/pipeline/data/backtest/`: resultados de backtests por ativo/fold/estratégia;
- `codigo/pipeline/models/`: modelos treinados;
- `codigo/pipeline/logs/`: logs de execução e histórico de treinamento.

## Reprodutibilidade

Para reproduzir resultados:

1. use o mesmo período e formato de dados definido no pipeline;
2. rode os comandos a partir de `codigo/pipeline/`;
3. execute preferencialmente com `uv` e ambiente Linux/WSL;
4. registre parâmetros, versões e logs das execuções.

## Documentação Técnica

Pontos de entrada mais úteis:

- `codigo/pipeline/README.md` (documentação operacional do pipeline);
- `codigo/pipeline/src/documentacao/INDICE.md` (índice técnico completo);
- `codigo/pipeline/src/documentacao/projeto/PROXIMOS_PASSOS_CONSOLIDADO.md` (planejamento e consolidação do projeto).

## Autor

Rafael da Silva Melo

## Licença

Uso acadêmico no contexto de Trabalho de Conclusão de Curso.
