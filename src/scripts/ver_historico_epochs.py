"""Script para visualizar histórico detalhado de epochs salvos em CSV."""

import sys
import pandas as pd
from pathlib import Path
from typing import Optional

def ver_historico_fold(ativo: str, modelo: str, fold: int) -> Optional[pd.DataFrame]:
    """
    Visualiza histórico de treinamento de um fold específico.
    
    Args:
        ativo: Nome do ativo (ex: VALE3)
        modelo: Tipo de modelo (ex: cnn_lstm)
        fold: Número do fold
    
    Returns:
        DataFrame com histórico ou None se não encontrado
    """
    csv_path = Path('logs') / 'training_history' / ativo / modelo / f'fold_{fold}_history.csv'
    
    if not csv_path.exists():
        print(f"❌ Histórico não encontrado: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    print("="*70)
    print(f"HISTÓRICO DE TREINAMENTO - Fold {fold}")
    print("="*70)
    print(f"Total de epochs: {len(df)}")
    print(f"Arquivo: {csv_path}")
    print("")
    
    # Mostrar primeiros e últimos epochs
    print("Primeiros 5 epochs:")
    print(df.head().to_string(index=False))
    print("")
    
    if len(df) > 10:
        print("Últimos 5 epochs:")
        print(df.tail().to_string(index=False))
        print("")
    
    # Estatísticas
    print("="*70)
    print("ESTATÍSTICAS")
    print("="*70)
    
    if 'val_loss' in df.columns:
        best_epoch = df['val_loss'].idxmin()
        print(f"Melhor epoch: {best_epoch + 1}")
        print(f"  val_loss: {df.loc[best_epoch, 'val_loss']:.6f}")
        print(f"  val_accuracy: {df.loc[best_epoch, 'val_accuracy']:.6f}")
        print(f"  loss: {df.loc[best_epoch, 'loss']:.6f}")
        print(f"  accuracy: {df.loc[best_epoch, 'accuracy']:.6f}")
    
    if 'lr' in df.columns:
        print(f"\nLearning rate:")
        print(f"  Inicial: {df['lr'].iloc[0]:.8f}")
        print(f"  Final: {df['lr'].iloc[-1]:.8f}")
        print(f"  Mínimo: {df['lr'].min():.8f}")
    
    print("")
    
    return df


def ver_todos_folds(ativo: str = "VALE3", modelo: str = "cnn_lstm", n_folds: int = 5):
    """
    Visualiza histórico de todos os folds.
    
    Args:
        ativo: Nome do ativo
        modelo: Tipo de modelo
        n_folds: Número de folds esperados
    """
    print("="*70)
    print("HISTÓRICO DE TREINAMENTO - TODOS OS FOLDS")
    print("="*70)
    print(f"Ativo: {ativo}")
    print(f"Modelo: {modelo.upper()}")
    print("")
    
    historicos = []
    
    for fold in range(1, n_folds + 1):
        csv_path = Path('logs') / 'training_history' / ativo / modelo / f'fold_{fold}_history.csv'
        
        if not csv_path.exists():
            print(f"Fold {fold}: ❌ Não encontrado")
            continue
        
        df = pd.read_csv(csv_path)
        
        # Estatísticas do fold
        total_epochs = len(df)
        
        if 'val_loss' in df.columns:
            best_epoch = df['val_loss'].idxmin()
            best_val_loss = df.loc[best_epoch, 'val_loss']
            best_val_acc = df.loc[best_epoch, 'val_accuracy']
            final_lr = df['lr'].iloc[-1] if 'lr' in df.columns else None
            
            historicos.append({
                'fold': fold,
                'epochs': total_epochs,
                'best_epoch': best_epoch + 1,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'final_lr': final_lr
            })
            
            print(f"Fold {fold}: ✅ {total_epochs} epochs, "
                  f"melhor epoch={best_epoch+1}, "
                  f"val_loss={best_val_loss:.4f}, "
                  f"val_acc={best_val_acc:.4f}")
        else:
            print(f"Fold {fold}: ✅ {total_epochs} epochs")
    
    if historicos:
        print("")
        print("="*70)
        print("RESUMO")
        print("="*70)
        df_resumo = pd.DataFrame(historicos)
        print(df_resumo.to_string(index=False))
        print("")
        print(f"Média de epochs: {df_resumo['epochs'].mean():.1f}")
        print(f"Média best_epoch: {df_resumo['best_epoch'].mean():.1f}")
        print(f"Média val_loss: {df_resumo['best_val_loss'].mean():.6f}")
        print(f"Média val_acc: {df_resumo['best_val_acc'].mean():.6f}")


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualizar histórico de treinamento (epochs salvos em CSV)'
    )
    parser.add_argument('--ativo', type=str, default='VALE3',
                       help='Nome do ativo')
    parser.add_argument('--modelo', type=str, default='cnn_lstm',
                       choices=['lstm', 'cnn_lstm'],
                       help='Tipo de modelo')
    parser.add_argument('--fold', type=int, default=None,
                       help='Número do fold (se não especificado, mostra todos)')
    
    args = parser.parse_args()
    
    if args.fold is not None:
        # Mostrar fold específico
        ver_historico_fold(args.ativo, args.modelo, args.fold)
    else:
        # Mostrar todos os folds
        ver_todos_folds(args.ativo, args.modelo)


if __name__ == '__main__':
    main()
