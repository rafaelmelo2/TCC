"""Script para fazer previsões em tempo real usando modelos salvos."""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Adicionar diretório pipeline ao path
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

import tensorflow as tf
from tensorflow import keras

from src.data_processing.load_data import carregar_dados
from src.data_processing.feature_engineering import criar_features
from src.data_processing.prepare_sequences import (
    selecionar_features_dl, criar_sequencias_temporais
)
from src.config import JANELA_TEMPORAL_STEPS
from sklearn.preprocessing import MinMaxScaler


def carregar_modelo(ativo: str, modelo_tipo: str, fold: Optional[int] = None) -> keras.Model:
    """Carrega modelo salvo."""
    models_dir = Path('models') / ativo / modelo_tipo
    
    if not models_dir.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {models_dir}")
    
    if fold is None:
        model_files = list(models_dir.glob('fold_*_checkpoint.keras'))
        if not model_files:
            raise FileNotFoundError(f"Nenhum modelo encontrado em: {models_dir}")
        fold = max([int(f.stem.split('_')[1]) for f in model_files])
    
    model_path = models_dir / f'fold_{fold}_checkpoint.keras'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    print(f"[OK] Carregando modelo: {model_path}")
    return keras.models.load_model(model_path)


def prever_proxima_vela(ativo: str, 
                        modelo_tipo: str = 'cnn_lstm',
                        arquivo_dados: Optional[str] = None,
                        fold: Optional[int] = None,
                        threshold: float = 0.5,
                        usar_ensemble: bool = False) -> Dict[str, Any]:
    """Faz previsão da próxima vela de 15 minutos."""
    print("="*70)
    print("PREVISÃO EM TEMPO REAL")
    print("="*70)
    print(f"Ativo: {ativo}")
    print(f"Modelo: {modelo_tipo.upper()}")
    print("")
    
    # Carregar dados
    if arquivo_dados is None:
        from src.config import obter_nome_arquivo_dados
        arquivo_dados = f'data/raw/{obter_nome_arquivo_dados(ativo)}'
    
    if not os.path.exists(arquivo_dados):
        raise FileNotFoundError(f"Arquivo não encontrado: {arquivo_dados}")
    
    print(f"[1/4] Carregando dados de {ativo}...")
    df = carregar_dados(arquivo_dados, verbose=False)
    print(f"[OK] {len(df)} barras carregadas")
    
    # Criar features
    print(f"[2/4] Criando features...")
    df_features = criar_features(df, verbose=False)
    print(f"[OK] Features criadas")
    
    # Verificar se há dados suficientes
    if len(df_features) < JANELA_TEMPORAL_STEPS:
        raise ValueError(
            f"[ERRO] Dados insuficientes. Necessário {JANELA_TEMPORAL_STEPS} barras, "
            f"mas apenas {len(df_features)} disponíveis"
        )
    
    # Preparar dados para previsão
    print(f"[3/4] Preparando sequência temporal (últimas {JANELA_TEMPORAL_STEPS} barras)...")
    feature_names = selecionar_features_dl(df_features)
    X_recent = df_features[feature_names].values[-JANELA_TEMPORAL_STEPS:]
    X_seq = X_recent.reshape(1, JANELA_TEMPORAL_STEPS, len(feature_names))
    
    # Normalizar (AVISO: em produção, use o scaler do treino!)
    scaler = MinMaxScaler()
    X_2d = X_seq.reshape(-1, len(feature_names))
    scaler.fit(X_2d)
    X_seq_norm_2d = scaler.transform(X_2d)
    X_seq_norm = X_seq_norm_2d.reshape(1, JANELA_TEMPORAL_STEPS, len(feature_names))
    print(f"[OK] Sequência preparada: {X_seq_norm.shape}")
    
    # Fazer previsão
    print(f"[4/4] Fazendo previsão...")
    
    if usar_ensemble:
        models_dir = Path('models') / ativo / modelo_tipo
        model_files = sorted(models_dir.glob('fold_*_checkpoint.keras'))
        
        if len(model_files) == 0:
            raise FileNotFoundError(f"Nenhum modelo encontrado em: {models_dir}")
        
        print(f"     Usando ensemble de {len(model_files)} modelos...")
        probabilidades = []
        
        for model_file in model_files:
            model = keras.models.load_model(model_file)
            proba = model.predict(X_seq_norm, verbose=0)[0][0]
            probabilidades.append(proba)
        
        probabilidade_media = np.mean(probabilidades)
        probabilidade_std = np.std(probabilidades)
        
        print(f"     Probabilidades: {[f'{p:.3f}' for p in probabilidades]}")
        print(f"     Média: {probabilidade_media:.3f} ± {probabilidade_std:.3f}")
        
        probabilidade = probabilidade_media
        
    else:
        model = carregar_modelo(ativo, modelo_tipo, fold)
        probabilidade = model.predict(X_seq_norm, verbose=0)[0][0]
    
    # Interpretar resultado
    direcao = 'ALTA' if probabilidade > threshold else 'BAIXA'
    
    distancia_threshold = abs(probabilidade - threshold)
    if distancia_threshold > 0.2:
        confianca = 'ALTA'
    elif distancia_threshold > 0.1:
        confianca = 'MEDIA'
    else:
        confianca = 'BAIXA'
    
    resultado = {
        'direcao': direcao,
        'probabilidade': float(probabilidade),
        'confianca': confianca,
        'ativo': ativo,
        'modelo': modelo_tipo,
        'threshold': threshold
    }
    
    # Exibir resultado
    print("")
    print("="*70)
    print("RESULTADO DA PREVISÃO")
    print("="*70)
    print(f"Direção Prevista: {direcao}")
    print(f"Probabilidade: {probabilidade:.3f} ({probabilidade*100:.1f}%)")
    print(f"Confiança: {confianca}")
    print("="*70)
    
    return resultado


def main():
    """Função principal com interface de linha de comando."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fazer previsão em tempo real usando modelo salvo'
    )
    parser.add_argument('--ativo', type=str, default='VALE3',
                       help='Nome do ativo (ex: VALE3, PETR4, ITUB4)')
    parser.add_argument('--modelo', type=str, default='cnn_lstm',
                       choices=['lstm', 'cnn_lstm'],
                       help='Tipo de modelo')
    parser.add_argument('--fold', type=int, default=None,
                       help='Número do fold a usar (None = mais recente)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Limiar para classificação (0.5 = probabilidade)')
    parser.add_argument('--ensemble', action='store_true',
                       help='Usar ensemble de todos os folds (média)')
    
    args = parser.parse_args()
    
    try:
        resultado = prever_proxima_vela(
            ativo=args.ativo,
            modelo_tipo=args.modelo,
            fold=args.fold,
            threshold=args.threshold,
            usar_ensemble=args.ensemble
        )
        
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
