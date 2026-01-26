"""Script para análise de modelos salvos do walk-forward."""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Adicionar diretório pipeline ao path
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import tensorflow as tf
from tensorflow import keras

from src.data_processing.load_data import carregar_dados
from src.data_processing.feature_engineering import criar_features
from src.data_processing.prepare_sequences import preparar_dados_dl
from src.utils.validation import WalkForwardValidator
from src.utils.metrics import calcular_metricas_preditivas, calcular_acuracia_direcional
from src.config import (
    TAMANHO_TREINO_BARRAS, TAMANHO_TESTE_BARRAS, EMBARGO_BARRAS,
    JANELA_TEMPORAL_STEPS, SEED
)


def carregar_modelo_fold(ativo: str, modelo_tipo: str, fold: int) -> keras.Model:
    """Carrega modelo salvo de um fold específico."""
    model_path = Path('models') / ativo / modelo_tipo / f'fold_{fold}_checkpoint.keras'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    return keras.models.load_model(model_path)


def analisar_modelo_fold(model: keras.Model, ativo: str, fold_num: int,
                         df_features: pd.DataFrame, fold_info: Any,
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Analisa modelo salvo em um fold específico.
    
    Parâmetros:
        model: Modelo carregado
        ativo: Nome do ativo
        fold_num: Número do fold
        df_features: DataFrame com features
        fold_info: Informações do fold (train/test indices)
        verbose: Se True, imprime informações
    
    Retorna:
        Dicionário com métricas e análises
    """
    # Preparar dados do fold
    X_train, y_train, X_test, y_test, scaler, _ = preparar_dados_dl(
        df_features,
        fold_info.train_start, fold_info.train_end,
        fold_info.test_start, fold_info.test_end,
        n_steps=JANELA_TEMPORAL_STEPS,
        verbose=False
    )
    
    # Fazer previsões
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.where(y_pred_proba > 0.5, 1, -1).flatten()
    
    # Calcular métricas
    metricas = calcular_metricas_preditivas(y_test, y_pred)
    metricas['accuracy_direcional'] = calcular_acuracia_direcional(y_test, y_pred)
    
    # Análise de probabilidades
    proba_stats = {
        'mean': float(np.mean(y_pred_proba)),
        'std': float(np.std(y_pred_proba)),
        'min': float(np.min(y_pred_proba)),
        'max': float(np.max(y_pred_proba)),
        'median': float(np.median(y_pred_proba))
    }
    
    # Análise de distribuição de previsões
    n_pred_alta = int(np.sum(y_pred == 1))
    n_pred_baixa = int(np.sum(y_pred == -1))
    n_real_alta = int(np.sum(y_test == 1))
    n_real_baixa = int(np.sum(y_test == -1))
    n_real_neutro = int(np.sum(y_test == 0))
    
    if verbose:
        print(f"\n[Fold {fold_num}] Análise do Modelo")
        print(f"  Previsões: Alta={n_pred_alta}, Baixa={n_pred_baixa}")
        print(f"  Real (teste): Alta={n_real_alta}, Baixa={n_real_baixa}, Neutro={n_real_neutro}")
        print(f"  Probabilidades: mean={proba_stats['mean']:.3f}, std={proba_stats['std']:.3f}")
        print(f"  Acurácia: {metricas['accuracy_direcional']:.4f}")
        print(f"  F1-Score: {metricas['f1_score']:.4f}")
        print(f"  MCC: {metricas['mcc']:.4f}")
    
    return {
        'fold': fold_num,
        'metricas': metricas,
        'proba_stats': proba_stats,
        'distribuicao_pred': {'alta': n_pred_alta, 'baixa': n_pred_baixa},
        'distribuicao_real': {'alta': n_real_alta, 'baixa': n_real_baixa, 'neutro': n_real_neutro},
        'n_train': fold_info.n_train,
        'n_test': fold_info.n_test
    }


def ensemble_previsoes(ativo: str, modelo_tipo: str, n_folds: int,
                       df_features: pd.DataFrame, folds_info: List[Any],
                       verbose: bool = True) -> pd.DataFrame:
    """
    Cria previsões ensemble combinando todos os folds.
    
    Estratégia: Média das probabilidades de todos os modelos.
    
    Parâmetros:
        ativo: Nome do ativo
        modelo_tipo: Tipo de modelo ('lstm' ou 'cnn_lstm')
        n_folds: Número de folds
        df_features: DataFrame com features
        folds_info: Lista de informações dos folds
        verbose: Se True, imprime informações
    
    Retorna:
        DataFrame com previsões ensemble e métricas
    """
    if verbose:
        print("\n" + "="*70)
        print("ENSEMBLE DE MODELOS")
        print("="*70)
    
    results_ensemble = []
    
    for i, fold in enumerate(folds_info):
        try:
            # Carregar modelo
            model = carregar_modelo_fold(ativo, modelo_tipo, i+1)
            
            # Preparar dados
            _, _, X_test, y_test, _, _ = preparar_dados_dl(
                df_features,
                fold.train_start, fold.train_end,
                fold.test_start, fold.test_end,
                n_steps=JANELA_TEMPORAL_STEPS,
                verbose=False
            )
            
            # Fazer previsões
            y_pred_proba = model.predict(X_test, verbose=0)
            
            # Para ensemble, vamos coletar probabilidades de cada modelo
            # e fazer média ponderada (pode ser implementado depois)
            y_pred = np.where(y_pred_proba > 0.5, 1, -1).flatten()
            
            # Calcular métricas
            metricas = calcular_metricas_preditivas(y_test, y_pred)
            metricas['accuracy_direcional'] = calcular_acuracia_direcional(y_test, y_pred)
            
            results_ensemble.append({
                'fold': i + 1,
                'accuracy_direcional': metricas['accuracy_direcional'],
                'f1_score': metricas['f1_score'],
                'mcc': metricas['mcc']
            })
            
            if verbose:
                print(f"  Fold {i+1}: Acurácia={metricas['accuracy_direcional']:.4f}, "
                      f"F1={metricas['f1_score']:.4f}, MCC={metricas['mcc']:.4f}")
        
        except FileNotFoundError as e:
            if verbose:
                print(f"  Fold {i+1}: Modelo não encontrado ({e})")
            continue
    
    if len(results_ensemble) == 0:
        raise FileNotFoundError("Nenhum modelo encontrado!")
    
    df_ensemble = pd.DataFrame(results_ensemble)
    
    if verbose:
        print(f"\n{'='*70}")
        print("RESULTADOS ENSEMBLE")
        print("="*70)
        print(f"Acurácia Média: {df_ensemble['accuracy_direcional'].mean():.4f}")
        print(f"F1-Score Médio: {df_ensemble['f1_score'].mean():.4f}")
        print(f"MCC Médio: {df_ensemble['mcc'].mean():.4f}")
    
    return df_ensemble


def main(ativo: str = "VALE3", modelo_tipo: str = "cnn_lstm",
         arquivo_dados: str = None, verbose: bool = True):
    """
    Analisa modelos salvos de um treinamento walk-forward.
    
    Parâmetros:
        ativo: Nome do ativo
        modelo_tipo: 'lstm' ou 'cnn_lstm'
        arquivo_dados: Caminho para arquivo de dados (opcional)
        verbose: Se True, imprime informações detalhadas
    """
    print("="*70)
    print("ANÁLISE DE MODELOS SALVOS")
    print("="*70)
    print(f"Ativo: {ativo}")
    print(f"Modelo: {modelo_tipo.upper()}")
    
    # Verificar se existem modelos salvos
    models_dir = Path('models') / ativo / modelo_tipo
    if not models_dir.exists():
        print(f"\n[ERRO] Diretório de modelos não encontrado: {models_dir}")
        print("Execute o treinamento primeiro com:")
        print(f"  uv run python src/train.py --ativo {ativo} --modelo {modelo_tipo} --optuna --n-trials 30")
        return
    
    # Listar modelos disponíveis
    model_files = list(models_dir.glob('fold_*_checkpoint.keras'))
    if len(model_files) == 0:
        print(f"\n[ERRO] Nenhum modelo encontrado em: {models_dir}")
        return
    
    print(f"\n[OK] {len(model_files)} modelos encontrados")
    for model_file in sorted(model_files):
        print(f"  - {model_file.name}")
    
    # Carregar dados
    if arquivo_dados is None:
        from src.config import obter_nome_arquivo_dados
        arquivo_dados = f'data/raw/{obter_nome_arquivo_dados(ativo)}'
    
    if not os.path.exists(arquivo_dados):
        print(f"\n[ERRO] Arquivo não encontrado: {arquivo_dados}")
        return
    
    print(f"\n[1/3] Carregando dados de {ativo}...")
    df = carregar_dados(arquivo_dados, verbose=False)
    df_features = criar_features(df, verbose=False)
    print(f"[OK] Dados carregados: {df_features.shape}")
    
    # Criar validador walk-forward
    print(f"\n[2/3] Configurando walk-forward validation...")
    validator = WalkForwardValidator(
        train_size=TAMANHO_TREINO_BARRAS,
        test_size=TAMANHO_TESTE_BARRAS,
        embargo=EMBARGO_BARRAS
    )
    folds_info = validator._gerar_folds(len(df_features))
    print(f"[OK] {len(folds_info)} folds configurados")
    
    # Analisar cada modelo
    print(f"\n[3/3] Analisando modelos salvos...")
    results = []
    
    for i, fold in enumerate(folds_info):
        try:
            model = carregar_modelo_fold(ativo, modelo_tipo, i+1)
            result = analisar_modelo_fold(
                model, ativo, i+1, df_features, fold, verbose=verbose
            )
            results.append(result)
        except FileNotFoundError as e:
            if verbose:
                print(f"\n[Fold {i+1}] Modelo não encontrado")
            continue
    
    if len(results) == 0:
        print("\n[ERRO] Nenhum modelo pôde ser analisado")
        return
    
    # Consolidar resultados
    print(f"\n{'='*70}")
    print("RESULTADOS CONSOLIDADOS")
    print("="*70)
    
    df_results = pd.DataFrame([{
        'fold': r['fold'],
        'accuracy_direcional': r['metricas']['accuracy_direcional'],
        'f1_score': r['metricas']['f1_score'],
        'mcc': r['metricas']['mcc'],
        'proba_mean': r['proba_stats']['mean'],
        'proba_std': r['proba_stats']['std']
    } for r in results])
    
    print("\nMétricas por Fold:")
    print(df_results.to_string(index=False))
    
    print(f"\nMétricas Agregadas:")
    print(f"  Acurácia Média: {df_results['accuracy_direcional'].mean():.4f} ± {df_results['accuracy_direcional'].std():.4f}")
    print(f"  F1-Score Médio: {df_results['f1_score'].mean():.4f} ± {df_results['f1_score'].std():.4f}")
    print(f"  MCC Médio: {df_results['mcc'].mean():.4f} ± {df_results['mcc'].std():.4f}")
    
    # Salvar análise
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    arquivo_analise = output_dir / f'{ativo}_{modelo_tipo}_analise_modelos.csv'
    df_results.to_csv(arquivo_analise, index=False)
    print(f"\n[OK] Análise salva em: {arquivo_analise}")
    
    print("\n" + "="*70)
    print("ANÁLISE CONCLUÍDA")
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analisar modelos salvos')
    parser.add_argument('--ativo', type=str, default='VALE3', help='Nome do ativo')
    parser.add_argument('--modelo', type=str, default='cnn_lstm', choices=['lstm', 'cnn_lstm'],
                       help='Tipo de modelo')
    parser.add_argument('--arquivo', type=str, default=None, help='Caminho para arquivo de dados')
    
    args = parser.parse_args()
    
    main(
        ativo=args.ativo,
        modelo_tipo=args.modelo,
        arquivo_dados=args.arquivo,
        verbose=True
    )
