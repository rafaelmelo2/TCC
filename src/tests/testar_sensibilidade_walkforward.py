"""
Script para análise de sensibilidade dos parâmetros de walk-forward validation.

Este script testa diferentes configurações de tamanho de treino, teste e embargo
para validar a robustez da configuração principal escolhida a priori.

METODOLOGIA:
- A configuração principal foi escolhida ANTES de executar experimentos
- Este script serve para VALIDAR robustez, não para escolher a melhor
- Resultados devem mostrar que a configuração principal é robusta
  (variações < 2-3% em acurácia direcional)

Conforme metodologia do TCC (Seção 4.4.3 - Análise de Sensibilidade).
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Suprimir avisos
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')

# Adicionar diretório pipeline ao path
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

from src.data_processing.load_data import carregar_dados
from src.data_processing.feature_engineering import criar_features
from src.models.baselines import NaiveBaseline, DriftBaseline
from src.utils.validation import WalkForwardValidator
from src.utils.metrics import calcular_metricas_preditivas, calcular_acuracia_direcional
from src.config import CONFIGURACOES_SENSIBILIDADE


def testar_configuracao(config: Dict[str, Any], df_features: pd.DataFrame, 
                        verbose: bool = True) -> Dict[str, Any]:
    """
    Testa uma configuração específica de walk-forward.
    
    Parâmetros:
        config: Dicionário com 'nome', 'treino', 'teste', 'embargo', 'descricao'
        df_features: DataFrame com features
        verbose: Se True, imprime progresso
    
    Retorna:
        Dicionário com resultados da configuração
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"TESTANDO: {config['nome'].upper()}")
        print(f"Descrição: {config['descricao']}")
        print(f"Treino: {config['treino']} barras | Teste: {config['teste']} barras | Embargo: {config['embargo']} barras")
        print(f"{'='*70}")
    
    # Criar validador com configuração específica
    validator = WalkForwardValidator(
        train_size=config['treino'],
        test_size=config['teste'],
        embargo=config['embargo']
    )
    
    # Gerar folds
    folds_info = validator._gerar_folds(len(df_features))
    
    if len(folds_info) == 0:
        if verbose:
            print(f"[ERRO] Não foi possível gerar folds para {config['nome']}")
        return {
            'configuracao': config['nome'],
            'n_folds': 0,
            'metricas': {}
        }
    
    if verbose:
        print(f"[OK] {len(folds_info)} folds gerados")
    
    # Testar com NaiveBaseline (modelo simples para validação rápida)
    baseline = NaiveBaseline()
    results = validator.validate(
        model=baseline,
        X=df_features,
        y=df_features['returns'],
        fit_func=lambda m, X, y: m.fit(y),
        predict_func=lambda m, X: m.predict(steps=len(X)),
        verbose=False
    )
    
    # Obter índices de teste
    test_indices = []
    for r in results['folds']:
        test_indices.extend(range(r['test_start'], r['test_end']))
    
    test_indices = [idx for idx in test_indices if idx < len(df_features)]
    y_true = df_features['target'].iloc[test_indices].values
    y_pred = results['all_y_pred']
    
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Calcular métricas
    metricas = calcular_metricas_preditivas(y_true, y_pred)
    metricas['accuracy_direcional'] = calcular_acuracia_direcional(y_true, y_pred)
    
    resultado = {
        'configuracao': config['nome'],
        'descricao': config['descricao'],
        'treino_barras': config['treino'],
        'teste_barras': config['teste'],
        'embargo_barras': config['embargo'],
        'n_folds': results['n_folds'],
        'total_test_samples': results['total_test_samples'],
        'metricas': metricas
    }
    
    if verbose:
        print(f"[OK] Concluído!")
        print(f"     Folds: {resultado['n_folds']} | Teste: {resultado['total_test_samples']}")
        print(f"     Acurácia Direcional: {metricas['accuracy_direcional']:.4f}")
    
    return resultado


def main(ativo: str = "VALE3", arquivo_dados: str = None, verbose: bool = True):
    """
    Executa análise de sensibilidade completa.
    
    Parâmetros:
        ativo: Nome do ativo (ex: "VALE3")
        arquivo_dados: Caminho para arquivo de dados (opcional)
        verbose: Se True, imprime informações detalhadas
    """
    print("=" * 70)
    print("ANÁLISE DE SENSIBILIDADE - WALK-FORWARD VALIDATION")
    print("=" * 70)
    print("\nOBJETIVO: Validar robustez da configuração principal")
    print("         (escolhida a priori, não otimizada)")
    
    # Carregar dados
    if arquivo_dados is None:
        from ..config import obter_nome_arquivo_dados
        arquivo_dados = f'data/raw/{obter_nome_arquivo_dados(ativo)}'
    
    if not os.path.exists(arquivo_dados):
        print(f"[ERRO] Arquivo não encontrado: {arquivo_dados}")
        return
    
    print(f"\n[1/3] Carregando dados de {ativo}...")
    df = carregar_dados(arquivo_dados, verbose=False)
    print(f"[OK] Dados carregados: {df.shape}")
    
    print(f"\n[2/3] Criando features...")
    df_features = criar_features(df, verbose=False)
    print(f"[OK] Features criadas: {df_features.shape}")
    
    print(f"\n[3/3] Testando {len(CONFIGURACOES_SENSIBILIDADE)} configurações...")
    
    # Testar todas as configurações
    resultados = []
    for config in CONFIGURACOES_SENSIBILIDADE:
        resultado = testar_configuracao(config, df_features, verbose=verbose)
        resultados.append(resultado)
    
    # Consolidar resultados
    print(f"\n{'='*70}")
    print("RESULTADOS CONSOLIDADOS")
    print(f"{'='*70}")
    
    df_resultados = pd.DataFrame([{
        'Configuração': r['configuracao'],
        'Descrição': r['descricao'],
        'Treino (barras)': r['treino_barras'],
        'Teste (barras)': r['teste_barras'],
        'Embargo (barras)': r['embargo_barras'],
        'N_Folds': r['n_folds'],
        'N_Teste': r['total_test_samples'],
        'Acurácia Direcional': r['metricas'].get('accuracy_direcional', np.nan),
        'Accuracy': r['metricas'].get('accuracy', np.nan),
        'F1-Score': r['metricas'].get('f1_score', np.nan),
        'MCC': r['metricas'].get('mcc', np.nan),
    } for r in resultados])
    
    print("\n" + df_resultados.to_string(index=False))
    
    # Análise comparativa
    print(f"\n{'='*70}")
    print("ANÁLISE COMPARATIVA")
    print(f"{'='*70}")
    
    # Encontrar configuração principal
    principal_idx = df_resultados[df_resultados['Configuração'] == 'principal'].index[0]
    principal_acc = df_resultados.loc[principal_idx, 'Acurácia Direcional']
    
    print(f"\nConfiguração Principal: {principal_acc:.4f}")
    print("\nComparação com outras configurações:")
    
    for idx, row in df_resultados.iterrows():
        if row['Configuração'] != 'principal':
            diff = row['Acurácia Direcional'] - principal_acc
            diff_pct = (diff / principal_acc) * 100 if principal_acc > 0 else 0
            print(f"  {row['Configuração']:20s}: {row['Acurácia Direcional']:.4f} "
                  f"({diff:+.4f}, {diff_pct:+.2f}%)")
    
    # Verificar robustez
    outras_accs = df_resultados[df_resultados['Configuração'] != 'principal']['Acurácia Direcional']
    max_diff = abs(outras_accs - principal_acc).max()
    max_diff_pct = (max_diff / principal_acc) * 100 if principal_acc > 0 else 0
    
    print(f"\n{'='*70}")
    print("CONCLUSÃO SOBRE ROBUSTEZ")
    print(f"{'='*70}")
    print(f"Diferença máxima em relação à configuração principal: {max_diff:.4f} ({max_diff_pct:.2f}%)")
    
    if max_diff_pct < 2:
        print("✅ Configuração principal é ROBUSTA (variação < 2%)")
    elif max_diff_pct < 5:
        print("⚠️  Configuração principal é ACEITÁVEL (variação < 5%)")
    else:
        print("❌ Configuração principal pode não ser robusta (variação >= 5%)")
        print("   Considere revisar a escolha ou investigar causas da variação")
    
    # Salvar resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('data/processed/walkforward')
    output_dir.mkdir(parents=True, exist_ok=True)
    arquivo_resultados = output_dir / f'{ativo}_sensibilidade_walkforward_{timestamp}.csv'
    df_resultados.to_csv(arquivo_resultados, index=False)
    print(f"\n[OK] Resultados salvos em: {arquivo_resultados}")
    
    print("\n" + "="*70)
    print("ANÁLISE DE SENSIBILIDADE CONCLUÍDA")
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Análise de sensibilidade dos parâmetros de walk-forward'
    )
    parser.add_argument('--ativo', type=str, default='VALE3', 
                       help='Nome do ativo')
    parser.add_argument('--arquivo', type=str, default=None, 
                       help='Caminho para arquivo de dados')
    parser.add_argument('--verbose', action='store_true', 
                       help='Imprimir informações detalhadas')
    
    args = parser.parse_args()
    
    main(
        ativo=args.ativo,
        arquivo_dados=args.arquivo,
        verbose=args.verbose
    )
