"""Gera tabelas e gráficos a partir dos resultados dos testes Diebold-Mariano.

Lê o CSV produzido por rodar_testes_estatisticos.py e gera:
- Tabela resumo: p-valores e significância (CNN-LSTM vs cada baseline)
- Tabela por regime (se houver coluna Regime)
- Gráfico: heatmap de p-valores ou barras de diferença de perda

Uso:
  uv run python src/scripts/gerar_tabelas_graficos_dm.py
  uv run python src/scripts/gerar_tabelas_graficos_dm.py --csv data/processed/testes_diebold_mariano.csv --saida_dir data/processed/dm_figuras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

script_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(script_dir))


def carregar_resultados(csv_path: str | Path) -> pd.DataFrame:
    """Carrega CSV de resultados DM."""
    df = pd.read_csv(csv_path)
    if "Regime" not in df.columns:
        df["Regime"] = "geral"
    return df


def tabela_resumo(df: pd.DataFrame, regime: str = "geral") -> pd.DataFrame:
    """Tabela Ativo x Baseline com DM_pvalue e estrelas de significância."""
    sub = df[df["Regime"] == regime] if "Regime" in df.columns else df
    if sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(
        index="Ativo",
        columns="Baseline",
        values="DM_pvalue",
        aggfunc="first",
    )
    # Estrelas: * 0.05, ** 0.01, *** 0.001
    def stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    try:
        pivot_str = pivot.map(lambda p: f"{p:.4f}" + stars(p) if pd.notna(p) else "")
    except AttributeError:
        pivot_str = pivot.applymap(lambda p: f"{p:.4f}" + stars(p) if pd.notna(p) else "")
    return pivot_str


def tabela_diferenca_perda(df: pd.DataFrame, regime: str = "geral") -> pd.DataFrame:
    """Tabela Ativo x Baseline com diferença de perda (CNN - baseline)."""
    sub = df[df["Regime"] == regime] if "Regime" in df.columns else df
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(
        index="Ativo",
        columns="Baseline",
        values="Diferenca_perda",
        aggfunc="first",
    )


def salvar_grafico_heatmap(df_pivot: pd.DataFrame, out_path: Path) -> None:
    """Salva heatmap de p-valores (se matplotlib disponível)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    if df_pivot.empty or df_pivot.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    # Valores numéricos para o heatmap (usar p-value)
    data = df_pivot.astype(float, errors="ignore")
    im = ax.imshow(np.where(np.isfinite(data), data, np.nan), aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.15)
    ax.set_xticks(range(len(df_pivot.columns)))
    ax.set_yticks(range(len(df_pivot.index)))
    ax.set_xticklabels(df_pivot.columns)
    ax.set_yticklabels(df_pivot.index)
    plt.colorbar(im, ax=ax, label="p-valor")
    ax.set_title("Teste Diebold-Mariano: p-valor (CNN-LSTM vs baseline)\n< 0.05 = diferença significativa")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Tabelas e gráficos dos testes Diebold-Mariano")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/processed/testes_diebold_mariano.csv",
        help="Caminho do CSV de resultados DM.",
    )
    parser.add_argument(
        "--saida_dir",
        type=str,
        default="data/processed",
        help="Diretório para salvar tabelas (CSV) e figuras.",
    )
    parser.add_argument(
        "--grafico",
        action="store_true",
        help="Gerar heatmap de p-valores (requer matplotlib).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {csv_path}")
        print("Execute antes: uv run python src/scripts/rodar_testes_estatisticos.py --todos")
        return

    out_dir = Path(args.saida_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = carregar_resultados(csv_path)
    if df.empty:
        print("[ERRO] CSV vazio ou sem colunas esperadas.")
        return

    print("RESULTADOS DIEBOLD-MARIANO - TABELAS")
    print("=" * 60)

    # Tabela resumo (só regime geral se existir)
    regimes = df["Regime"].unique().tolist() if "Regime" in df.columns else ["geral"]
    for reg in regimes:
        tab = tabela_resumo(df, regime=reg)
        if tab.empty:
            continue
        suf = f"_regime_{reg}" if reg != "geral" else ""
        path_tab = out_dir / f"dm_resumo_pvalores{suf}.csv"
        tab.to_csv(path_tab)
        print(f"\n[Regime: {reg}] P-valores (e significância *<0.05 **<0.01 ***<0.001):")
        print(tab.to_string())
        print(f"Salvo: {path_tab}")

    # Tabela diferença de perda (geral)
    tab_diff = tabela_diferenca_perda(df, regime="geral")
    if not tab_diff.empty:
        path_diff = out_dir / "dm_diferenca_perda_geral.csv"
        tab_diff.to_csv(path_diff)
        print(f"\nDiferença de perda (CNN - baseline); negativo = CNN melhor:")
        print(tab_diff.round(4).to_string())
        print(f"Salvo: {path_diff}")

    if args.grafico:
        # Heatmap só para regime geral (p-valores numéricos)
        sub = df[df["Regime"] == "geral"] if "Regime" in df.columns else df
        if not sub.empty and "DM_pvalue" in sub.columns:
            pivot_num = sub.pivot_table(index="Ativo", columns="Baseline", values="DM_pvalue", aggfunc="first")
            path_fig = out_dir / "dm_heatmap_pvalores.png"
            salvar_grafico_heatmap(pivot_num, path_fig)
            if path_fig.exists():
                print(f"\nFigura salva: {path_fig}")

    print("\nConcluído.")


if __name__ == "__main__":
    main()
