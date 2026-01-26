"""Script para teste rápido de validação das melhorias implementadas."""

import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List

def analisar_resultados_teste(ativo: str = "VALE3", modelo: str = "cnn_lstm") -> Dict:
    """
    Analisa resultados do teste rápido e decide se deve prosseguir.
    
    Retorna:
        dict com 'aprovado' (bool) e 'motivos' (list)
    """
    # Ler resultados
    results_file = Path(f'data/processed/{ativo}_{modelo}_walkforward.csv')
    
    if not results_file.exists():
        return {
            'aprovado': False,
            'motivos': [f"Arquivo de resultados não encontrado: {results_file}"]
        }
    
    df = pd.read_csv(results_file)
    
    # Critérios de aprovação
    criterios = []
    aprovado = True
    
    print("="*70)
    print("ANÁLISE DO TESTE RÁPIDO")
    print("="*70)
    print(f"\nResultados encontrados: {len(df)} folds")
    print(df.to_string(index=False))
    
    # Critério 1: Acurácia média > 51%
    acc_media = df['accuracy_direcional'].mean()
    if acc_media > 0.51:
        criterios.append(f"✅ Acurácia média: {acc_media:.4f} > 0.51")
    else:
        criterios.append(f"❌ Acurácia média: {acc_media:.4f} <= 0.51")
        aprovado = False
    
    # Critério 2: Primeiro fold melhor que antes
    if len(df) >= 1:
        fold1_acc = df.iloc[0]['accuracy_direcional']
        baseline_fold1 = 0.4687  # Resultado anterior
        if fold1_acc > baseline_fold1:
            criterios.append(f"✅ Fold 1: {fold1_acc:.4f} > {baseline_fold1:.4f} (baseline)")
        else:
            criterios.append(f"⚠️ Fold 1: {fold1_acc:.4f} <= {baseline_fold1:.4f} (baseline)")
            # Não reprovar só por isso, pode ser variação
    
    # Critério 3: MCC médio > 0.04
    if 'mcc' in df.columns:
        mcc_medio = df['mcc'].mean()
        baseline_mcc = 0.039
        if mcc_medio > baseline_mcc:
            criterios.append(f"✅ MCC médio: {mcc_medio:.4f} > {baseline_mcc:.4f} (baseline)")
        else:
            criterios.append(f"⚠️ MCC médio: {mcc_medio:.4f} <= {baseline_mcc:.4f} (baseline)")
    
    # Critério 4: F1-Score razoável
    if 'f1_score' in df.columns:
        f1_medio = df['f1_score'].mean()
        if f1_medio > 0.55:
            criterios.append(f"✅ F1-Score médio: {f1_medio:.4f} > 0.55")
        else:
            criterios.append(f"⚠️ F1-Score médio: {f1_medio:.4f} <= 0.55")
    
    # Critério 5: Variabilidade não muito alta
    acc_std = df['accuracy_direcional'].std()
    if acc_std < 0.10:  # Menos de 10 pontos percentuais
        criterios.append(f"✅ Variabilidade: {acc_std:.4f} < 0.10 (estável)")
    else:
        criterios.append(f"⚠️ Variabilidade: {acc_std:.4f} >= 0.10 (alta)")
    
    # Verificar se modelos foram salvos
    models_dir = Path('models') / ativo / modelo
    if models_dir.exists():
        model_files = list(models_dir.glob('fold_*_checkpoint.keras'))
        if len(model_files) >= len(df):
            criterios.append(f"✅ Modelos salvos: {len(model_files)} arquivos")
        else:
            criterios.append(f"❌ Modelos salvos: {len(model_files)} < {len(df)} esperados")
            aprovado = False
    else:
        criterios.append(f"❌ Diretório de modelos não encontrado: {models_dir}")
        aprovado = False
    
    print(f"\n{'='*70}")
    print("CRITÉRIOS DE APROVAÇÃO")
    print("="*70)
    for criterio in criterios:
        print(f"  {criterio}")
    
    print(f"\n{'='*70}")
    if aprovado:
        print("✅ TESTE APROVADO - Pode prosseguir com treinamento completo!")
        print("="*70)
        print("\nComando para treinamento completo (rodar à noite):")
        print(f"\nuv run python src/train.py \\")
        print(f"    --ativo {ativo} \\")
        print(f"    --modelo {modelo} \\")
        print(f"    --optuna \\")
        print(f"    --n-trials 50 \\")
        print(f"    --epochs 150")
    else:
        print("❌ TESTE REPROVADO - Necessário investigar e corrigir")
        print("="*70)
        print("\nProblemas encontrados:")
        for criterio in criterios:
            if '❌' in criterio:
                print(f"  - {criterio}")
    
    print("\n")
    
    return {
        'aprovado': aprovado,
        'criterios': criterios,
        'metricas': {
            'accuracy_media': float(acc_media),
            'accuracy_std': float(acc_std),
            'mcc_medio': float(df['mcc'].mean()) if 'mcc' in df.columns else None,
            'f1_medio': float(df['f1_score'].mean()) if 'f1_score' in df.columns else None
        }
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analisar teste rápido')
    parser.add_argument('--ativo', type=str, default='VALE3')
    parser.add_argument('--modelo', type=str, default='cnn_lstm')
    
    args = parser.parse_args()
    
    resultado = analisar_resultados_teste(args.ativo, args.modelo)
    
    # Exit code 0 se aprovado, 1 se reprovado
    sys.exit(0 if resultado['aprovado'] else 1)
