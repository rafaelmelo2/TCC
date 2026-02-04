# Backtesting com Custos de Transação

**Referência TCC:** Seção 4.5.1 – Backtests e Custos de Transação  

Sobre testes estatísticos (Diebold-Mariano, TCC 4.5.2): ver [testes_estatisticos_diebold_mariano.md](testes_estatisticos_diebold_mariano.md).  
**Módulo:** `src/utils/backtesting.py`  
**Script de uso:** `src/scripts/rodar_backtest.py`  
**Data:** 2026-02-02

---

## 1. Objetivo

Avaliar a **utilidade prática** das previsões do modelo sob custos realistas (corretagem, emolumentos, slippage), em conformidade com as práticas da B3. O TCC exige:

- Backtests **long-only** e **long/short** condicionados às probabilidades previstas e a limiares calibrados.
- Custos fixos e proporcionais e slippage descontados.
- Relato de **turnover** e **sensibilidade a custos** (análise de estresse).

---

## 2. Métricas Calculadas

| Métrica | Descrição | Fórmula / convenção |
|--------|-----------|----------------------|
| **Retorno líquido** | Variação percentual do capital após custos | (capital_final - capital_inicial) / capital_inicial |
| **Sharpe ratio** | Retorno excedente por unidade de risco (anualizado) | (média dos retornos líquidos / desvio) × √(barras_por_ano) |
| **Max drawdown** | Maior queda percentual do pico ao vale da curva de capital | min((equity - pico_acumulado) / pico_acumulado) |
| **Profit factor** | Razão ganhos/perdas | soma(retornos positivos) / \|soma(retornos negativos)\| |
| **Turnover** | Frequência de mudança de posição | número de mudanças de posição / número de barras |
| **N. operações** | Número de “pernas” (0→1 = 1, 1→-1 = 2) | usado para custo fixo por operação |
| **Custo total (R$)** | Soma dos custos pagos no backtest | fixo + proporcional + slippage |

- **Barras por ano:** 26 barras/dia × 252 dias ≈ 6552 (barras de 15 min, pregão B3).

---

## 3. Custos de Transação

Configuração padrão (em `src/config.py`), atualizada em fev/2025:

- **Custo fixo (corretagem):** R$ 0,00 por operação. Corretagem zero é o padrão de mercado (Clear, entre outras); XP cobra R$ 2,90 (day trade) e R$ 4,90 (swing). Valores como R$ 10/operação estão desatualizados.
- **Custo proporcional (B3):** 0,03% do valor negociado. Corresponde às tarifas B3 para operações regulares (Negociação + CCP + TTA = 0,0300%); em day trade a B3 pode ser menor (ex.: 0,023% na primeira faixa de ADTV).
- **Slippage:** 0,05% do valor negociado. Faixa realista para ações líquidas é 0,05%–0,1%; 0,01% tende a subestimar custos.

Custos são aplicados **apenas quando a posição muda** (entrada ou saída). Round-trip 1→-1 conta como duas operações (fechar long, abrir short).

---

## 4. Estratégias

- **long_only:** sinal -1 (baixa) vira posição 0 (neutro). Apenas compra ou neutro.
- **long_short:** sinal 1 = long, -1 = short, 0 = neutro.

Sinal a partir de probabilidade: acima de `limiar_alta` → 1; abaixo de `(1 - limiar_baixa)` → -1; entre os dois → 0 (neutro). Padrão: limiar 0,5 para ambos.

---

## 5. Convenção Temporal

- **signal[i]** = decisão no início do período i.
- **returns_realized[i]** = retorno logarítmico realizado no período i (o que se ganharia estando long).
- P&L do período: **posição[i] × returns_realized[i]** (para short, posição = -1).

No pipeline walk-forward, a previsão na amostra de teste j refere-se ao retorno na barra **seguinte** à janela; portanto `returns_realized[j]` deve ser o retorno na barra `test_start + n_steps + j + 1`. O helper `retornos_e_sinal_para_backtest()` faz esse alinhamento.

---

## 6. Uso do Módulo

### 6.1 Uso direto (arrays)

```python
from src.utils.backtesting import (
    CustosBacktest,
    run_backtest,
    sinal_de_probabilidade,
)

# Retornos realizados (log) e sinal 1 / -1 / 0
returns_realized = ...  # shape (N,)
signal = ...            # shape (N,)

custos = CustosBacktest.from_config()  # ou CustosBacktest(custo_fixo=10, ...)

resultado = run_backtest(
    returns_realized=returns_realized,
    signal=signal,
    custos=custos,
    estrategia="long_short",
)

# resultado["retorno_liquido"], resultado["sharpe_ratio"], resultado["max_drawdown"], etc.
```

### 6.2 Sinal a partir de probabilidade

```python
from src.utils.backtesting import sinal_de_probabilidade

proba = model.predict(X_test).flatten()  # P(alta)
signal = sinal_de_probabilidade(proba, limiar_alta=0.5, limiar_baixa=0.5)
```

### 6.3 Alinhamento com walk-forward

```python
from src.utils.backtesting import retornos_e_sinal_para_backtest

returns = df_features["returns"].values
returns_realized, signal = retornos_e_sinal_para_backtest(
    returns, signal, fold_info.test_start, n_steps=60,
)
```

### 6.4 Sensibilidade a custos (estresse)

```python
from src.utils.backtesting import run_backtest_sensibilidade_custos

cenarios = run_backtest_sensibilidade_custos(
    returns_realized=returns_realized,
    signal=signal,
    custos_base=custos,
    estrategia="long_short",
    multiplicadores_custo=[0.5, 1.0, 1.5, 2.0],
)
# Cada item em cenarios tem as mesmas chaves de run_backtest + "multiplicador_custo"
```

---

## 7. Script de linha de comando e salvamento

### Onde os resultados são salvos

Todos os resultados são gravados em **`data/backtest/`**. A pasta é criada automaticamente na primeira execução.

- **Um arquivo por execução:**  
  `{ATIVO}_fold{FOLD}_{ESTRATEGIA}_{AAAAMMDD}_{HHMMSS}.csv`  
  Exemplo: `VALE3_fold1_long_short_20260202_183112.csv`

- **Com sensibilidade a custos:**  
  Além do arquivo acima, é gerado:  
  `{ATIVO}_fold{FOLD}_{ESTRATEGIA}_sensibilidade_{AAAAMMDD}_{HHMMSS}.csv`  
  Exemplo: `PETR4_fold2_long_only_sensibilidade_20260202_183121.csv`

- **Histórico consolidado:**  
  `historico_backtest.csv` — uma linha por execução (append), para comparar runs ao longo do tempo.

Detalhes da estrutura e das colunas estão em **`data/backtest/README.md`**.

### Comandos

```bash
# Backtest padrão (VALE3, fold 1, long_short)
uv run python src/scripts/rodar_backtest.py --ativo VALE3

# Outro ativo e fold
uv run python src/scripts/rodar_backtest.py --ativo PETR4 --fold 2

# Long-only
uv run python src/scripts/rodar_backtest.py --ativo ITUB4 --estrategia long_only

# Incluir análise de sensibilidade a custos
uv run python src/scripts/rodar_backtest.py --ativo VALE3 --sensibilidade
```

O script carrega dados e features, o modelo do fold indicado, gera previsões, alinha retornos com `retornos_e_sinal_para_backtest`, chama `run_backtest` (e opcionalmente `run_backtest_sensibilidade_custos`) e **salva automaticamente** em `data/backtest/` com o nome detalhado acima.

---

## 8. Fontes (custos de transação)

Usadas para calibrar `CUSTO_CORRETAGEM`, `CUSTO_TAXA_PROPORCIONAL` e `CUSTO_SLIPPAGE` no backtest (fev/2025).

| Parâmetro | Fonte | URL |
|-----------|--------|-----|
| **Corretagem** | Clear Corretora — corretagem zero; XP — valores por produto | [Clear – Custos](https://www.clear.com.br/site/custos/precos); [Corretora Clear – Custos](https://corretora.clear.com.br/custos/) |
| **Taxa proporcional (B3)** | B3 — Tarifas de Ações e Fundos de Investimento (à vista) | [B3 – Tarifas Ações à vista](https://www.b3.com.br/pt_br/produtos-e-servicos/tarifas/listados-a-vista-e-derivativos/renda-variavel/tarifas-de-acoes-e-fundos-de-investimento/a-vista/) |
| **Slippage** | Prática de mercado e regras do domínio (0,05%–0,1%) | `.cursor/rules/dominio-financeiro.mdc` |

**Resumo B3 (operações regulares, ADTV até R$ 3 mi):** Negociação 0,00500%, CCP 0,02240%, TTA 0,0026% → **total 0,0300%**. Day trade: tabela progressiva (ex.: 0,023% a 0,0115% conforme ADTV). Consultar a página da B3 para valores atualizados.

---

## 9. Referências

- TCC: Seção 4.5.1 (Backtests e Custos de Transação).
- SHARPE, W. F. The sharpe ratio. *Journal of Portfolio Management*, 1994.
- Regras do domínio: `.cursor/rules/dominio-financeiro.mdc` (custos B3, backtests).
