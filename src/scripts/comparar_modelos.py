"""
Gera tabela comparativa CNN-LSTM vs baselines (Naive, Drift, ARIMA, Prophet).

Conforme TCC Seção 4: comparação do modelo proposto com baselines em validação
walk-forward. Consolida resultados de:
  - data/processed/{ATIVO}_baselines_walkforward.csv
  - data/processed/{ATIVO}_cnn_lstm_analise_modelos.csv (ou _cnn_lstm_walkforward.csv)

Uso:
  uv run python src/scripts/comparar_modelos.py
  uv run python src/scripts/comparar_modelos.py --saida data/processed/comparativo_tcc.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Permitir execução a partir do diretório do pipeline
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ATIVOS = ('PETR4', 'VALE3', 'ITUB4')
DIR_PROCESSED = ROOT / 'data' / 'processed'


def _carregar_baselines() -> pd.DataFrame:
    """Carrega e concatena todos os CSVs de baselines por ativo."""
    dfs = []
    for ativo in ATIVOS:
        path = DIR_PROCESSED / f'{ativo}_baselines_walkforward.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _carregar_cnn_lstm_por_ativo() -> pd.DataFrame:
    """Para cada ativo, carrega métricas agregadas do CNN-LSTM (analise ou walkforward)."""
    rows = []
    for ativo in ATIVOS:
        analise = DIR_PROCESSED / f'{ativo}_cnn_lstm_analise_modelos.csv'
        walkf = DIR_PROCESSED / f'{ativo}_cnn_lstm_walkforward.csv'
        if analise.exists():
            df = pd.read_csv(analise)
        elif walkf.exists():
            df = pd.read_csv(walkf)
        else:
            continue
        # Agregar por ativo (média dos folds)
        acc_dir = df['accuracy_direcional'].mean()
        row = {
            'Ativo': ativo,
            'Modelo': 'CNN-LSTM',
            'N_Folds': len(df),
            'N_Teste': None,
            'accuracy': df['accuracy'].mean() if 'accuracy' in df.columns else acc_dir,
            'balanced_accuracy': df['balanced_accuracy'].mean() if 'balanced_accuracy' in df.columns else None,
            'f1_score': df['f1_score'].mean(),
            'mcc': df['mcc'].mean(),
            'accuracy_direcional': acc_dir,
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


COLS_COMPARATIVO = [
    'Ativo', 'Modelo', 'N_Folds', 'N_Teste',
    'accuracy', 'balanced_accuracy', 'f1_score', 'mcc', 'accuracy_direcional'
]


def main(saida: Path | None = None, verbose: bool = True) -> pd.DataFrame:
    saida = saida or DIR_PROCESSED / 'comparativo_cnn_lstm_vs_baselines.csv'
    saida = Path(saida)

    df_b = _carregar_baselines()
    df_c = _carregar_cnn_lstm_por_ativo()

    if df_b.empty and df_c.empty:
        if verbose:
            print('[ERRO] Nenhum arquivo de baselines ou CNN-LSTM encontrado em data/processed/')
        return pd.DataFrame()

    # Baselines: renomear Baseline -> Modelo e padronizar colunas
    if not df_b.empty:
        tb = df_b.rename(columns={'Baseline': 'Modelo'})
        for c in COLS_COMPARATIVO:
            if c not in tb.columns:
                tb[c] = pd.NA
        tb = tb[[c for c in COLS_COMPARATIVO if c in tb.columns]]
    else:
        tb = pd.DataFrame(columns=COLS_COMPARATIVO)

    # CNN-LSTM: garantir mesmas colunas e juntar à tabela
    if not df_c.empty:
        tc = df_c.copy()
        tc['Modelo'] = 'CNN-LSTM'
        if 'N_Teste' not in tc.columns:
            tc['N_Teste'] = pd.NA
        for c in COLS_COMPARATIVO:
            if c not in tc.columns:
                tc[c] = pd.NA
        tc = tc.reindex(columns=COLS_COMPARATIVO)
        tb = pd.concat([tb, tc], ignore_index=True) if not tb.empty else tc.reset_index(drop=True)

    # Ordem: por Ativo, depois CNN-LSTM primeiro e depois baselines
    ordem_modelo = ['CNN-LSTM', 'Naive', 'Drift', 'ARIMA', 'Prophet']
    tb['_ordem'] = tb['Modelo'].map(lambda m: ordem_modelo.index(m) if m in ordem_modelo else 99)
    tb = tb.sort_values(['Ativo', '_ordem']).drop(columns=['_ordem'], errors='ignore')

    # Arredondar para exibição
    for c in ('accuracy', 'balanced_accuracy', 'f1_score', 'mcc', 'accuracy_direcional'):
        if c in tb.columns:
            tb[c] = tb[c].round(4)

    tb.to_csv(saida, index=False)
    if verbose:
        print('=' * 70)
        print('COMPARATIVO CNN-LSTM vs BASELINES')
        print('=' * 70)
        print(f'\nArquivo gerado: {saida}')
        print('\nTabela consolidada (Ativo | Modelo | accuracy_direcional | f1_score | mcc):')
        print(tb.to_string(index=False))
        print('\n' + '=' * 70)
        print('MELHOR MODELO POR ATIVO (accuracy_direcional)')
        print('=' * 70)
        for ativo in ATIVOS:
            sub = tb[tb['Ativo'] == ativo]
            if sub.empty:
                continue
            idx = sub['accuracy_direcional'].idxmax()
            melhor = sub.loc[idx]
            print(f"  {ativo}: {melhor['Modelo']} ({100 * melhor['accuracy_direcional']:.2f}%)")
        print('=' * 70)
    return tb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gera tabela comparativa CNN-LSTM vs baselines')
    parser.add_argument('--saida', type=str, default=None, help='Caminho do CSV de saída')
    parser.add_argument('--quiet', action='store_true', help='Não imprimir resumo')
    args = parser.parse_args()
    main(saida=Path(args.saida) if args.saida else None, verbose=not args.quiet)
