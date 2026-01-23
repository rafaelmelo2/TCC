# Pipeline ARIMA - Previsão de Preços de Ações

Este pipeline implementa um modelo ARIMA simples para prever preços de ações da B3.

> 🚀 **Início Rápido**: Veja [INICIO_RAPIDO.md](INICIO_RAPIDO.md) para começar em 3 comandos!

## 📋 O que é ARIMA?

ARIMA significa **AutoRegressive Integrated Moving Average** (Média Móvel Integrada Autoregressiva).

É um modelo estatístico usado para prever valores futuros em séries temporais. Funciona assim:

- **AR (AutoRegressivo)**: usa valores passados para prever o futuro
- **I (Integrado)**: torna a série "estacionária" (remove tendências)
- **MA (Média Móvel)**: usa erros de previsão passados

## 🚀 Como Usar

### 1. Instalar uv (se ainda não tiver)

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Ou via pip
pip install uv
```

### 2. Instalar dependências do projeto

```bash
cd pipeline
uv sync
```

### 3. Executar o exemplo simples

```bash
uv run exemplo_simples_arima.py
```

**Alternativa rápida** (sem instalar):

```bash
uv run --with pandas --with numpy --with statsmodels --with matplotlib --with scikit-learn exemplo_simples_arima.py
```

Este script vai:

1. ✅ Carregar os dados do ativo ITUB4
2. ✅ Preparar a série temporal
3. ✅ Dividir em treino (80%) e teste (20%)
4. ✅ Treinar o modelo ARIMA
5. ✅ Fazer previsões
6. ✅ Avaliar o desempenho
7. ✅ Salvar resultados em CSV
8. ✅ Criar gráfico comparativo

### 3. Trocar de ativo

No arquivo `exemplo_simples_arima.py`, mude a linha:

```python
ATIVO = "ITUB4"  # Pode mudar para PETR4 ou VALE3
```

## 📁 Estrutura do Projeto

```
pipeline/
├── data/
│   ├── raw/              # Dados brutos (CSV dos ativos)
│   └── processed/        # Resultados e gráficos
│
├── src/
│   ├── data_processing/  # Funções para carregar dados
│   ├── models/           # Modelo ARIMA
│   └── utils/            # Visualizações
│
├── exemplo_simples_arima.py  # Script principal (COMECE POR AQUI!)
├── pyproject.toml            # Configuração do projeto (uv)
├── requirements.txt          # Dependências (compatibilidade pip)
└── README.md                 # Este arquivo
```

## 🎯 O que cada arquivo faz?

### `src/data_processing/load_data.py`

Funções simples para:

- Carregar dados do CSV
- Preparar série temporal
- Dividir em treino e teste

### `src/models/arima_model.py`

Funções para:

- Treinar modelo ARIMA
- Fazer previsões
- Avaliar desempenho

### `exemplo_simples_arima.py`

Script completo que executa todo o pipeline de forma clara e explicada.

## 📊 Resultados

Após executar, você terá:

1. **CSV com resultados**: `data/processed/{ATIVO}_resultados_arima.csv`

   - Data, valor real, valor previsto e erro

2. **Gráfico**: `data/processed/{ATIVO}_grafico_arima.png`
   - Comparação visual entre valores reais e previstos

## 🔧 Ajustar Parâmetros do ARIMA

No `exemplo_simples_arima.py`, você pode mudar:

```python
p = 2  # quantos valores passados usar (teste: 1, 2, 3, 5)
d = 1  # quantas diferenças fazer (teste: 0, 1, 2)
q = 2  # tamanho da média móvel (teste: 1, 2, 3, 5)
```

**Dica**: Comece com valores pequenos (1 ou 2) e vá testando!

## 📈 Entendendo as Métricas

- **MAE (Erro Médio)**: Quanto erramos em média (em R$)

  - Quanto menor, melhor!

- **RMSE**: Similar ao MAE, mas penaliza erros grandes

  - Quanto menor, melhor!

- **MAPE (Erro %)**: Erro em porcentagem
  - Exemplo: 2% significa que erramos 2% do valor real em média
  - Quanto menor, melhor!

## ❓ Próximos Passos

1. Execute o script com diferentes ativos (ITUB4, PETR4, VALE3)
2. Teste diferentes parâmetros (p, d, q)
3. Compare os resultados e veja qual combinação funciona melhor
4. Observe os gráficos para entender onde o modelo erra mais

## 🚀 Por que usar UV?

O **uv** é um gerenciador de pacotes Python moderno:

- ⚡ **10-100x mais rápido** que pip
- 🎯 Um comando só: `uv run script.py`
- 🔒 Gerencia ambientes virtuais automaticamente
- 📦 Compatível com pip/PyPI

**Veja mais detalhes**: [GUIA_UV.md](GUIA_UV.md)

## 📚 Para Aprender Mais

- ARIMA funciona melhor com séries estacionárias
- Períodos muito voláteis são mais difíceis de prever
- É normal ter algum erro - nenhum modelo é perfeito!
- Compare diferentes modelos para ver qual performa melhor

## 🔄 Compatibilidade com PIP

Se preferir usar pip tradicional, ainda funciona:

```bash
pip install -r requirements.txt
python exemplo_simples_arima.py
```
