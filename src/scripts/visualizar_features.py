"""
Script para visualização gráfica das features criadas.

Gera visualizações separadas por ano e por tipo de indicador, organizadas em:
  data/visualizacoes/{ATIVO}/{ANO}/
    - EMA_{ano}.png          (Preços + Médias Móveis Exponenciais)
    - bollinger_{ano}.png    (Bandas de Bollinger)
    - RSI_{ano}.png          (Relative Strength Index)
    - retornos_{ano}.png     (Retornos logarítmicos)
    - volatilidade_{ano}.png
    - target_{ano}.png       (Target com banda morta)
    - distribuicoes_{ano}.png
    - correlacoes_{ano}.png
    - estatisticas_target_{ano}.png

Conforme metodologia do TCC (Seção 4.2 - Engenharia de Features).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend não interativo para salvar muitos PNGs
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List

# Adicionar diretório pipeline ao path
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

PROJECT_ROOT = script_dir

from src.data_processing.load_data import carregar_dados
from src.data_processing.feature_engineering import criar_features
from src.config import (
    PERIODOS_EMA, PERIODOS_RSI, PERIODO_BOLLINGER,
    THRESHOLD_BANDA_MORTA, obter_nome_arquivo_dados
)

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")

# Colunas a excluir das features técnicas (OHLCV, metadata, target)
COLUNAS_EXCLUIR = [
    'abertura', 'maxima', 'minima', 'fechamento', 'volume_real', 'target',
    'ticker', 'simbolo_mt5', 'volume_ticks', 'spread', 'data'
]


def _dir_visualizacoes(ativo: str, ano: int) -> Path:
    """Retorna o diretório de saída: data/visualizacoes/{ativo}/{ano}/"""
    d = PROJECT_ROOT / 'data' / 'visualizacoes' / ativo / str(ano)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _features_tecnicas(df: pd.DataFrame) -> List[str]:
    """Lista de colunas de features técnicas (exclui OHLCV, metadata, target)."""
    return [c for c in df.columns if c not in COLUNAS_EXCLUIR]


def _plot_ema(df: pd.DataFrame, ativo: str, ano: int, output_dir: Path, salvar: bool) -> None:
    """Preços + EMAs (Médias Móveis Exponenciais). Salva EMA_{ano}.png."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['fechamento'], label='Fechamento', linewidth=1.5, color='black', alpha=0.8)
    cores = ['blue', 'orange', 'red']
    for i, p in enumerate(PERIODOS_EMA):
        col = f'ema_{p}'
        if col in df.columns:
            ax.plot(df.index, df[col], label=f'EMA {p}', linewidth=1.5, color=cores[i], alpha=0.7)
    ax.set_title(f'{ativo} - Preços e EMAs ({ano})', fontsize=14, fontweight='bold')
    ax.set_ylabel('Preço (R$)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if salvar:
        png = output_dir / f'EMA_{ano}.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"    [OK] {png.name}")
    plt.close()


def _plot_bollinger(df: pd.DataFrame, ativo: str, ano: int, output_dir: Path, salvar: bool) -> None:
    """Bandas de Bollinger + preço. Salva bollinger_{ano}.png."""
    if 'bb_upper' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['fechamento'], label='Fechamento', linewidth=1.5, color='black', alpha=0.8)
    ax.fill_between(df.index, df['bb_lower'], df['bb_upper'], alpha=0.2, color='gray', label='Bandas Bollinger')
    ax.plot(df.index, df['bb_middle'], label='BB Middle', linewidth=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'{ativo} - Bandas de Bollinger ({ano})', fontsize=14, fontweight='bold')
    ax.set_ylabel('Preço (R$)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if salvar:
        png = output_dir / f'bollinger_{ano}.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"    [OK] {png.name}")
    plt.close()


def _plot_rsi(df: pd.DataFrame, ativo: str, ano: int, output_dir: Path, salvar: bool) -> None:
    """RSI. Salva RSI_{ano}.png."""
    fig, ax = plt.subplots(figsize=(14, 4))
    cores = ['purple', 'green', 'brown']
    for i, p in enumerate(PERIODOS_RSI):
        col = f'rsi_{p}'
        if col in df.columns:
            ax.plot(df.index, df[col], label=f'RSI {p}', linewidth=1.5, color=cores[i], alpha=0.7)
    ax.axhline(70, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(30, color='g', linestyle='--', alpha=0.5, linewidth=1)
    ax.fill_between(df.index, 70, 100, alpha=0.1, color='red')
    ax.fill_between(df.index, 0, 30, alpha=0.1, color='green')
    ax.set_ylabel('RSI', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title(f'{ativo} - RSI ({ano})', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if salvar:
        png = output_dir / f'RSI_{ano}.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"    [OK] {png.name}")
    plt.close()


def _plot_retornos(df: pd.DataFrame, ativo: str, ano: int, output_dir: Path, salvar: bool) -> None:
    """Retornos logarítmicos + banda morta. Salva retornos_{ano}.png."""
    if 'returns' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df['returns'] * 100, label='Retornos log.', linewidth=1, color='blue', alpha=0.6)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.axhline(THRESHOLD_BANDA_MORTA * 100, color='red', linestyle='--', alpha=0.5, linewidth=1,
               label=f'Banda morta (±{THRESHOLD_BANDA_MORTA*100:.2f}%)')
    ax.axhline(-THRESHOLD_BANDA_MORTA * 100, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylabel('Retornos (%)', fontsize=12)
    ax.set_title(f'{ativo} - Retornos logarítmicos ({ano})', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if salvar:
        png = output_dir / f'retornos_{ano}.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"    [OK] {png.name}")
    plt.close()


def _plot_volatilidade(df: pd.DataFrame, ativo: str, ano: int, output_dir: Path, salvar: bool) -> None:
    """Volatilidade. Salva volatilidade_{ano}.png."""
    if 'volatility' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df['volatility'] * 100, label='Volatilidade', linewidth=1.5, color='orange', alpha=0.7)
    ax.set_ylabel('Volatilidade (%)', fontsize=12)
    ax.set_title(f'{ativo} - Volatilidade ({ano})', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if salvar:
        png = output_dir / f'volatilidade_{ano}.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"    [OK] {png.name}")
    plt.close()


def _plot_target(df: pd.DataFrame, ativo: str, ano: int, output_dir: Path, salvar: bool) -> None:
    """Target (Alta/Baixa/Neutro). Salva target_{ano}.png."""
    if 'target' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(14, 4))
    alta = df[df['target'] == 1]
    baixa = df[df['target'] == -1]
    neutro = df[df['target'] == 0]
    if len(alta):
        ax.scatter(alta.index, alta['target'], color='green', marker='^', s=20, alpha=0.6, label='Alta (1)')
    if len(baixa):
        ax.scatter(baixa.index, baixa['target'], color='red', marker='v', s=20, alpha=0.6, label='Baixa (-1)')
    if len(neutro):
        ax.scatter(neutro.index, neutro['target'], color='gray', marker='o', s=10, alpha=0.3, label='Neutro (0)')
    ax.set_ylabel('Target', fontsize=12)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f'{ativo} - Target com banda morta ({ano})', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if salvar:
        png = output_dir / f'target_{ano}.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"    [OK] {png.name}")
    plt.close()


def _plot_distribuicoes(df: pd.DataFrame, ativo: str, ano: int, output_dir: Path, salvar: bool) -> None:
    """Distribuições das features. Salva distribuicoes_{ano}.png."""
    feats = _features_tecnicas(df)
    if not feats:
        return
    n = len(feats)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.atleast_2d(axes)
    for i, col in enumerate(feats):
        r, c = i // ncols, i % ncols
        ax = axes[r, c]
        d = df[col].dropna()
        if len(d):
            ax.hist(d, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(col, fontsize=11, fontweight='bold')
            ax.set_xlabel('Valor')
            ax.set_ylabel('Frequência')
            ax.axvline(d.mean(), color='red', linestyle='--', linewidth=2, label=f'μ={d.mean():.4f}')
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for j in range(len(feats), nrows * ncols):
        fig.delaxes(axes.flatten()[j])
    fig.suptitle(f'{ativo} - Distribuições das features ({ano})', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    if salvar:
        png = output_dir / f'distribuicoes_{ano}.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"    [OK] {png.name}")
    plt.close()


def _plot_correlacoes(df: pd.DataFrame, ativo: str, ano: int, output_dir: Path, salvar: bool) -> None:
    """Matriz de correlação. Salva correlacoes_{ano}.png."""
    feats = _features_tecnicas(df)
    if not feats:
        return
    corr = df[feats].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True,
                linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title(f'{ativo} - Correlações ({ano})', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if salvar:
        png = output_dir / f'correlacoes_{ano}.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"    [OK] {png.name}")
    plt.close()


def _plot_estatisticas_target(df: pd.DataFrame, ativo: str, ano: int, output_dir: Path, salvar: bool) -> None:
    """Distribuição de classes e retornos por classe. Salva estatisticas_target_{ano}.png."""
    if 'target' not in df.columns:
        return
    target = df['target'].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Barras
    cnt = target.value_counts().sort_index()
    cores = {-1: 'red', 0: 'gray', 1: 'green'}
    labs = {-1: 'Baixa (-1)', 0: 'Neutro (0)', 1: 'Alta (1)'}
    axes[0].bar([labs[k] for k in cnt.index], cnt.values, color=[cores[k] for k in cnt.index], alpha=0.7, edgecolor='black')
    for b in axes[0].patches:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2, h, f'{int(h)}\n({h/len(target)*100:.1f}%)',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].set_title(f'Distribuição de classes ({ano})', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequência')
    axes[0].grid(True, alpha=0.3, axis='y')
    # Boxplot retornos por classe
    if 'returns' in df.columns:
        ret = df['returns'].dropna()
        tgt = target.loc[ret.index]
        dados = [ret[tgt == -1].values * 100, ret[tgt == 0].values * 100, ret[tgt == 1].values * 100]
        bp = axes[1].boxplot(dados, labels=['Baixa (-1)', 'Neutro (0)', 'Alta (1)'], patch_artist=True)
        for patch, cor in zip(bp['boxes'], ['red', 'gray', 'green']):
            patch.set_facecolor(cor)
            patch.set_alpha(0.7)
        axes[1].axhline(0, color='black', linestyle='-', alpha=0.3)
        axes[1].axhline(THRESHOLD_BANDA_MORTA * 100, color='red', linestyle='--', alpha=0.5)
        axes[1].axhline(-THRESHOLD_BANDA_MORTA * 100, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title(f'Retornos por classe ({ano})', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Retornos (%)')
    axes[1].grid(True, alpha=0.3, axis='y')
    fig.suptitle(f'{ativo} - Estatísticas do target', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if salvar:
        png = output_dir / f'estatisticas_target_{ano}.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"    [OK] {png.name}")
    plt.close()


def _gerar_visualizacoes_ano(df_ano: pd.DataFrame, ativo: str, ano: int, salvar: bool) -> None:
    """Gera todos os gráficos para um ano e salva em data/visualizacoes/{ativo}/{ano}/."""
    out = _dir_visualizacoes(ativo, ano)
    _plot_ema(df_ano, ativo, ano, out, salvar)
    _plot_bollinger(df_ano, ativo, ano, out, salvar)
    _plot_rsi(df_ano, ativo, ano, out, salvar)
    _plot_retornos(df_ano, ativo, ano, out, salvar)
    _plot_volatilidade(df_ano, ativo, ano, out, salvar)
    _plot_target(df_ano, ativo, ano, out, salvar)
    _plot_distribuicoes(df_ano, ativo, ano, out, salvar)
    _plot_correlacoes(df_ano, ativo, ano, out, salvar)
    _plot_estatisticas_target(df_ano, ativo, ano, out, salvar)


def imprimir_estatisticas_descritivas(df_features: pd.DataFrame, ativo: str) -> None:
    """Imprime estatísticas descritivas das features."""
    print("\n" + "="*70)
    print(f"ESTATÍSTICAS DESCRITIVAS - {ativo}")
    print("="*70)
    feats = _features_tecnicas(df_features)
    if not feats:
        print("[AVISO] Nenhuma feature técnica encontrada.")
        return
    print("\nEstatísticas básicas:")
    print(df_features[feats].describe().round(4))
    if 'target' in df_features.columns:
        t = df_features['target'].dropna()
        cnt = t.value_counts().sort_index()
        total = len(t)
        print("\n" + "-"*70)
        print("DISTRIBUIÇÃO DO TARGET:")
        print("-"*70)
        for k, v in cnt.items():
            lab = {-1: 'Baixa (-1)', 0: 'Neutro (0)', 1: 'Alta (1)'}[k]
            print(f"  {lab}: {v:>6} ({v/total*100:>5.2f}%)")
        print(f"  Total: {total:>6}")
        print(f"  Threshold banda morta: ±{THRESHOLD_BANDA_MORTA*100:.2f}%")


def main(ativo: str = "VALE3", arquivo_dados: Optional[str] = None,
         salvar_graficos: bool = True, verbose: bool = True) -> None:
    """
    Carrega dados, cria features e gera visualizações por ano em
    data/visualizacoes/{ativo}/{ano}/, com gráficos separados (EMA, RSI, etc.).
    """
    print("="*70)
    print("VISUALIZAÇÃO DE FEATURES (por ano)")
    print("="*70)
    print(f"Ativo: {ativo}")

    if arquivo_dados is None:
        arquivo_dados = PROJECT_ROOT / 'data' / 'raw' / obter_nome_arquivo_dados(ativo)
    else:
        arquivo_dados = Path(arquivo_dados)
        if not arquivo_dados.is_absolute():
            arquivo_dados = PROJECT_ROOT / arquivo_dados

    if not arquivo_dados.exists():
        print(f"\n[ERRO] Arquivo não encontrado: {arquivo_dados}")
        raw_dir = PROJECT_ROOT / 'data' / 'raw'
        if raw_dir.exists():
            print("Arquivos em data/raw/:")
            for f in raw_dir.glob('*.csv'):
                print(f"  - {f.name}")
        return

    arquivo_dados = str(arquivo_dados)

    print("\n[1/4] Carregando dados...")
    df = carregar_dados(arquivo_dados, verbose=verbose)
    print(f"[OK] Dados: {df.shape}")

    print("\n[2/4] Criando features...")
    df_features = criar_features(df, verbose=verbose)
    print(f"[OK] Features: {df_features.shape}")

    print("\n[3/4] Estatísticas descritivas...")
    imprimir_estatisticas_descritivas(df_features, ativo)

    anos = sorted(df_features.index.year.unique())
    print(f"\n[4/4] Gerando visualizações por ano: {anos}")

    base_out = PROJECT_ROOT / 'data' / 'visualizacoes' / ativo
    base_out.mkdir(parents=True, exist_ok=True)
    for ano in anos:
        (base_out / str(ano)).mkdir(parents=True, exist_ok=True)

    for ano in anos:
        df_ano = df_features[df_features.index.year == ano]
        if df_ano.empty:
            continue
        print(f"\n  → {ativo} / {ano} ({len(df_ano)} barras)")
        _gerar_visualizacoes_ano(df_ano, ativo, ano, salvar_graficos)

    print("\n" + "="*70)
    print("VISUALIZAÇÃO CONCLUÍDA")
    print("="*70)
    if salvar_graficos:
        print(f"Gráficos em: data/visualizacoes/{ativo}/")
        print("Estrutura: data/visualizacoes/{ativo}/{ano}/EMA_{ano}.png, RSI_{ano}.png, ...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualização de features por ano (pastas ativo/ano, gráficos separados)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  uv run python src/scripts/visualizar_features.py --ativo VALE3
  uv run python src/scripts/visualizar_features.py --ativo PETR4 --no-salvar
        """
    )
    parser.add_argument('--ativo', type=str, default='VALE3', choices=['VALE3', 'PETR4', 'ITUB4'])
    parser.add_argument('--arquivo-dados', type=str, default=None)
    parser.add_argument('--salvar', action='store_true', default=True)
    parser.add_argument('--no-salvar', dest='salvar', action='store_false')
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--quiet', dest='verbose', action='store_false')
    args = parser.parse_args()

    main(ativo=args.ativo, arquivo_dados=args.arquivo_dados,
         salvar_graficos=args.salvar, verbose=args.verbose)
