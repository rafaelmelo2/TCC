"""Script para testar baselines com validação walk-forward.

Conforme TCC Seção 4: comparar modelo proposto (CNN-LSTM) com baselines
Naive/Drift, ARIMA e Prophet em validação walk-forward.

Uso:
  uv run python src/tests/testar_baselines_walkforward.py --ativo PETR4
  uv run python src/tests/testar_baselines_walkforward.py --ativo VALE3
  uv run python src/tests/testar_baselines_walkforward.py --ativo ITUB4
  uv run python src/tests/testar_baselines_walkforward.py --todos  # os 3 ativos
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# Suprimir avisos do statsmodels sobre frequência de data
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')

# Adicionar diretório pipeline ao path para imports relativos funcionarem
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

from src.data_processing.load_data import carregar_dados
from src.data_processing.feature_engineering import criar_features
from src.models.baselines import NaiveBaseline, DriftBaseline, ARIMABaseline
from src.models.prophet_model import ProphetBaseline
from src.utils.validation import WalkForwardValidator
from src.utils.metrics import calcular_metricas_preditivas, calcular_acuracia_direcional
from src.config import (
    TAMANHO_TREINO_BARRAS,
    TAMANHO_TESTE_BARRAS,
    EMBARGO_BARRAS,
    obter_nome_arquivo_dados,
)


def testar_baseline(baseline, nome, df_features, validator):
    """Testa um baseline com walk-forward."""
    print(f"\n{'='*70}")
    print(f"TESTANDO {nome.upper()}")
    print(f"{'='*70}")
    
    results = validator.validate(
        model=baseline,
        X=df_features,
        y=df_features['returns'],
        fit_func=lambda m, X, y: m.fit(y),
        predict_func=lambda m, X: m.predict(steps=len(X)) if hasattr(m, 'predict') else m.predict(),
        verbose=True
    )
    
    test_indices = []
    for r in results['folds']:
        test_indices.extend(range(r['test_start'], r['test_end']))
    
    test_indices = [idx for idx in test_indices if idx < len(df_features)]
    y_true = df_features['target'].iloc[test_indices].values
    y_pred = results['all_y_pred']
    
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    metricas = calcular_metricas_preditivas(y_true, y_pred)
    metricas['accuracy_direcional'] = calcular_acuracia_direcional(y_true, y_pred)
    
    results['metricas_finais'] = metricas
    
    print(f"\n[OK] {nome} concluído!")
    print(f"     Folds: {results['n_folds']} | Teste: {results['total_test_samples']}")
    print(f"     Acurácia Direcional: {metricas['accuracy_direcional']:.4f}")
    
    return results


def rodar_para_ativo(ativo: str) -> bool:
    """Executa teste de baselines walk-forward para um ativo. Retorna True se ok."""
    arquivo = f'data/raw/{obter_nome_arquivo_dados(ativo)}'

    if not os.path.exists(arquivo):
        print(f"[ERRO] Arquivo não encontrado: {arquivo}")
        return False

    print(f"\n[1/4] Carregando dados de {ativo}...")
    df = carregar_dados(arquivo, verbose=False)
    print(f"[OK] Dados carregados: {df.shape}")

    print(f"\n[2/4] Criando features...")
    df_features = criar_features(df, verbose=False)
    print(f"[OK] Features criadas: {df_features.shape}")

    if len(df_features) < TAMANHO_TREINO_BARRAS + TAMANHO_TESTE_BARRAS + EMBARGO_BARRAS:
        train_size = min(1000, len(df_features) // 3)
        test_size = min(100, len(df_features) // 10)
    else:
        train_size = TAMANHO_TREINO_BARRAS
        test_size = TAMANHO_TESTE_BARRAS

    print(f"\n[3/4] Criando validador (Treino: {train_size}, Teste: {test_size}, Embargo: {EMBARGO_BARRAS})...")
    validator = WalkForwardValidator(train_size=train_size, test_size=test_size, embargo=EMBARGO_BARRAS)
    folds_info = validator._gerar_folds(len(df_features))
    print(f"[OK] {len(folds_info)} folds serão gerados")

    print(f"\n[4/4] Testando baselines...")
    resultados = {}

    resultados['Naive'] = testar_baseline(NaiveBaseline(), 'Naive', df_features, validator)
    resultados['Drift'] = testar_baseline(DriftBaseline(), 'Drift', df_features, validator)

    try:
        resultados['ARIMA'] = testar_baseline(ARIMABaseline(max_p=2, max_d=1, max_q=2), 'ARIMA', df_features, validator)
    except Exception as e:
        print(f"[!] ARIMA falhou: {e}")

    try:
        resultados['Prophet'] = testar_baseline(ProphetBaseline(), 'Prophet', df_features, validator)
    except Exception as e:
        print(f"[!] Prophet falhou: {e}")

    comparacao = []
    for nome, res in resultados.items():
        if res and 'metricas_finais' in res:
            row = {'Ativo': ativo, 'Baseline': nome, 'N_Folds': res['n_folds'], 'N_Teste': res['total_test_samples']}
            row.update(res['metricas_finais'])
            comparacao.append(row)

    if comparacao:
        df_comparacao = pd.DataFrame(comparacao)
        output_dir = Path('data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)
        arquivo_resultados = output_dir / f'{ativo}_baselines_walkforward.csv'
        df_comparacao.to_csv(arquivo_resultados, index=False)
        print(f"\n[OK] Resultados salvos em: {arquivo_resultados}")
        print("\n" + "=" * 70)
        print(f"RESULTADOS {ativo}")
        print("=" * 70)
        print(df_comparacao.to_string(index=False))

        print("\n" + "=" * 70)
        print("MELHORES BASELINES POR MÉTRICA")
        print("=" * 70)
        for metrica in ['accuracy_direcional', 'accuracy', 'f1_score', 'mcc']:
            if metrica in df_comparacao.columns:
                melhor = df_comparacao.loc[df_comparacao[metrica].idxmax()]
                print(f"{metrica.upper()}: {melhor['Baseline']} ({melhor[metrica]:.4f})")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Testa baselines (Naive, Drift, ARIMA, Prophet) com validação walk-forward.'
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--ativo', choices=['PETR4', 'VALE3', 'ITUB4'], help='Ativo a testar')
    g.add_argument('--todos', action='store_true', help='Executar para os 3 ativos (PETR4, VALE3, ITUB4)')
    args = parser.parse_args()

    if args.todos:
        ativos = ['PETR4', 'VALE3', 'ITUB4']
    else:
        ativos = [args.ativo]

    print("=" * 70)
    print("TESTE DE BASELINES COM WALK-FORWARD VALIDATION")
    print("=" * 70)

    for i, ativo in enumerate(ativos, 1):
        if len(ativos) > 1:
            print(f"\n{'#'*70}")
            print(f"# ATIVO {i}/{len(ativos)}: {ativo}")
            print(f"{'#'*70}")
        rodar_para_ativo(ativo)

    print("\n" + "=" * 70)
    print("TESTE CONCLUÍDO")
    print("=" * 70)


if __name__ == '__main__':
    main()
