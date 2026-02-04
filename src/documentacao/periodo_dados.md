# Período Exato dos Dados

**Data:** 2025-01-23  
**Status:** Referência (config em `src/config.py`)

---

## Informação Crítica

**IMPORTANTE**: Os arquivos de dados estão nomeados como `{TICKER}_M15_20200101_20251231.csv`, mas os dados reais **não começam em 01/01/2020**.

## Período Real dos Dados

- **Data de início**: 22 de outubro de 2020, 10:00 (primeira barra disponível)
- **Data de fim**: 22 de outubro de 2025, 17:00 (5 anos completos a partir do início)
- **Duração**: Exatamente 5 anos de dados intradiários

## Por que essa diferença?

Os dados foram baixados do MetaTrader 5, que possui histórico limitado para barras de 15 minutos. A primeira barra disponível para os ativos utilizados (PETR4, VALE3, ITUB4) é de **22/10/2020 às 10:00**.

## Configuração no Código

As datas exatas estão definidas em `src/config.py`:

```python
DATA_INICIO_DADOS = datetime(2020, 10, 22)  # Primeira barra disponível
DATA_FIM_DADOS = datetime(2025, 10, 22)      # 5 anos completos
```

## Nomes dos Arquivos

Os arquivos mantêm o nome antigo (`20200101_20251231`) por questões de:
- **Compatibilidade**: Scripts e referências já existentes
- **Conveniência**: Não precisar renomear arquivos grandes
- **Clareza**: O nome indica o período desejado, não o período real

## Uso no Código

Para obter o nome correto do arquivo, use a função helper:

```python
from src.config import obter_nome_arquivo_dados

arquivo = f'data/raw/{obter_nome_arquivo_dados("PETR4")}'
# Retorna: 'PETR4_M15_20200101_20251231.csv'
```

## Validação

Ao carregar os dados, o sistema automaticamente:
1. Carrega apenas as barras disponíveis (a partir de 22/10/2020)
2. Valida que os dados estão dentro do período esperado
3. Reporta o período real nos logs

Exemplo de saída:
```
[OK] Shape final: (36482, 8) | Período: 2020-10-22 10:00:00 até 2025-10-22 17:00:00
```

## Impacto na Metodologia

Esta limitação **não afeta** a metodologia do TCC:
- ✅ 5 anos completos de dados (objetivo alcançado)
- ✅ Período suficiente para walk-forward validation
- ✅ Dados cobrem múltiplos regimes de mercado
- ✅ Granularidade de 15 minutos mantida

## Referências

- Arquivos de dados: `data/raw/`
- Configuração: `src/config.py`
- Função helper: `obter_nome_arquivo_dados()`

---

**Última atualização**: 23/01/2025  
**Documentado por**: Descoberta durante verificação de histórico no MetaTrader 5
