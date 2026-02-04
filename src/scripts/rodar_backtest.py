"""
Script para rodar backtest com custos sobre previsões do CNN-LSTM (walk-forward).

Conforme TCC Seção 4.5.1: backtests long-only e long/short com custos e slippage,
reportando retorno líquido, Sharpe, max drawdown, profit factor e turnover.

Resultados são salvos em data/backtest/ com nome detalhado:
  {ativo}_fold{fold}_{estrategia}_{AAAAMMDD}_{HHMMSS}.csv
  e append em data/backtest/historico_backtest.csv

Uso:
  uv run python src/scripts/rodar_backtest.py --ativo VALE3
  uv run python src/scripts/rodar_backtest.py --ativo PETR4 --fold 1 --estrategia long_short
  uv run python src/scripts/rodar_backtest.py --ativo ITUB4 --sensibilidade
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Garantir que o pipeline está no path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pasta onde os resultados de backtest são salvos (data/backtest/)
DIR_BACKTEST = ROOT / "data" / "backtest"
ARQUIVO_HISTORICO = DIR_BACKTEST / "historico_backtest.csv"

from src.config import (
    EMBARGO_BARRAS,
    JANELA_TEMPORAL_STEPS,
    TAMANHO_TESTE_BARRAS,
    TAMANHO_TREINO_BARRAS,
)
from src.data_processing.load_data import carregar_dados
from src.data_processing.feature_engineering import criar_features
from src.data_processing.prepare_sequences import preparar_dados_dl
from src.utils.validation import WalkForwardValidator
from src.utils.backtesting import (
    CustosBacktest,
    run_backtest,
    run_backtest_sensibilidade_custos,
    sinal_de_probabilidade,
    retornos_e_sinal_para_backtest,
)


def _carregar_modelo_fold(ativo: str, fold: int):
    """Carrega modelo Keras do fold (com custom_objects para focal loss)."""
    from tensorflow import keras
    from src.utils.focal_loss import focal_loss

    model_path = Path("models") / ativo / "cnn_lstm" / f"fold_{fold}_checkpoint.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    custom_objects = {"focal_loss_fixed": focal_loss(gamma=5.0, alpha=0.5)}
    return keras.models.load_model(model_path, custom_objects=custom_objects)


def _backtest_fold(
    ativo: str,
    fold: int,
    df_features,
    fold_info,
    estrategia: str = "long_short",
    custos: CustosBacktest | None = None,
    sensibilidade: bool = False,
):
    """
    Roda backtest para um fold: carrega modelo, previsões, alinha retornos e executa.

    Retorna dicionário com resultado do backtest (e opcionalmente lista de sensibilidade).
    """
    if custos is None:
        custos = CustosBacktest.from_config()

    model = _carregar_modelo_fold(ativo, fold)
    X_train, y_train, X_test, y_test, scaler, _ = preparar_dados_dl(
        df_features,
        fold_info.train_start, fold_info.train_end,
        fold_info.test_start, fold_info.test_end,
        n_steps=JANELA_TEMPORAL_STEPS,
        verbose=False,
    )
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    signal = sinal_de_probabilidade(y_pred_proba, limiar_alta=0.5, limiar_baixa=0.5)
    returns = df_features["returns"].values
    returns_realized, signal = retornos_e_sinal_para_backtest(
        returns, signal, fold_info.test_start, JANELA_TEMPORAL_STEPS,
    )
    if len(returns_realized) == 0:
        return {"erro": "Retornos/sinal vazios após alinhamento"}

    resultado = run_backtest(
        returns_realized=returns_realized,
        signal=signal,
        custos=custos,
        estrategia=estrategia,
    )
    resultado["ativo"] = ativo
    resultado["fold"] = fold
    resultado["estrategia"] = estrategia
    resultado["n_barras"] = len(returns_realized)
    resultado["capital_inicial"] = custos.capital_inicial

    if sensibilidade:
        resultados_sens = run_backtest_sensibilidade_custos(
            returns_realized=returns_realized,
            signal=signal,
            custos_base=custos,
            estrategia=estrategia,
            multiplicadores_custo=[0.5, 1.0, 1.5, 2.0],
        )
        for r in resultados_sens:
            r["ativo"] = ativo
            r["fold"] = fold
            r["estrategia"] = estrategia
        resultado["sensibilidade"] = resultados_sens

    return resultado


def _linha_resumo(resultado: dict, data_hora_str: str) -> dict:
    """Monta uma linha (dict) com métricas escalares para salvar em CSV."""
    return {
        "data_hora": data_hora_str,
        "ativo": resultado["ativo"],
        "fold": resultado["fold"],
        "estrategia": resultado["estrategia"],
        "retorno_liquido": resultado["retorno_liquido"],
        "sharpe_ratio": resultado["sharpe_ratio"],
        "max_drawdown": resultado["max_drawdown"],
        "profit_factor": resultado["profit_factor"],
        "turnover": resultado["turnover"],
        "n_trades": resultado["n_trades"],
        "custo_total_reais": resultado["custo_total_reais"],
        "capital_inicial": resultado.get("capital_inicial"),
        "capital_final": resultado["capital_final"],
        "n_barras": resultado["n_barras"],
    }


def _salvar_resultado(
    resultado: dict,
    data_hora_str: str,
    sensibilidade: bool,
) -> list[Path]:
    """
    Salva resultado do backtest em data/backtest/ com nome detalhado.

    - Um CSV por execução: {ativo}_fold{fold}_{estrategia}_{AAAAMMDD}_{HHMMSS}.csv
    - Append em historico_backtest.csv
    - Se sensibilidade: {ativo}_fold{fold}_{estrategia}_sensibilidade_{AAAAMMDD}_{HHMMSS}.csv

    Retorna lista de arquivos salvos.
    """
    if "erro" in resultado:
        return []

    DIR_BACKTEST.mkdir(parents=True, exist_ok=True)
    ativo = resultado["ativo"]
    fold = resultado["fold"]
    estrategia = resultado["estrategia"]
    salvos = []

    # Linha resumo (sem arrays)
    row = _linha_resumo(resultado, data_hora_str)
    df_run = pd.DataFrame([row])

    # Arquivo desta execução
    nome_run = f"{ativo}_fold{fold}_{estrategia}_{data_hora_str}.csv"
    path_run = DIR_BACKTEST / nome_run
    df_run.to_csv(path_run, index=False)
    salvos.append(path_run)

    # Append no histórico consolidado
    if ARQUIVO_HISTORICO.exists():
        df_hist = pd.read_csv(ARQUIVO_HISTORICO)
        df_hist = pd.concat([df_hist, df_run], ignore_index=True)
    else:
        df_hist = df_run
    df_hist.to_csv(ARQUIVO_HISTORICO, index=False)
    salvos.append(ARQUIVO_HISTORICO)

    # Sensibilidade a custos (tabela separada)
    if sensibilidade and "sensibilidade" in resultado:
        linhas = []
        for s in resultado["sensibilidade"]:
            linhas.append({
                "data_hora": data_hora_str,
                "ativo": ativo,
                "fold": fold,
                "estrategia": estrategia,
                "multiplicador_custo": s["multiplicador_custo"],
                "retorno_liquido": s["retorno_liquido"],
                "sharpe_ratio": s["sharpe_ratio"],
                "max_drawdown": s["max_drawdown"],
                "profit_factor": s["profit_factor"],
                "turnover": s["turnover"],
                "n_trades": s["n_trades"],
                "custo_total_reais": s["custo_total_reais"],
                "capital_final": s["capital_final"],
            })
        df_sens = pd.DataFrame(linhas)
        nome_sens = f"{ativo}_fold{fold}_{estrategia}_sensibilidade_{data_hora_str}.csv"
        path_sens = DIR_BACKTEST / nome_sens
        df_sens.to_csv(path_sens, index=False)
        salvos.append(path_sens)

    return salvos


def _imprimir_resultado(r: dict, sensibilidade: bool = False):
    """Imprime resultado do backtest de forma legível."""
    if "erro" in r:
        print(f"  [ERRO] {r['erro']}")
        return
    print(f"  Retorno líquido:    {r['retorno_liquido']:.4f} ({100 * r['retorno_liquido']:.2f}%)")
    print(f"  Sharpe ratio:      {r['sharpe_ratio']:.4f}")
    print(f"  Max drawdown:      {r['max_drawdown']:.4f} ({100 * r['max_drawdown']:.2f}%)")
    print(f"  Profit factor:     {r['profit_factor']:.4f}")
    print(f"  Turnover:          {r['turnover']:.4f}")
    print(f"  N. operações:      {r['n_trades']}")
    print(f"  Custo total (R$):  {r['custo_total_reais']:.2f}")
    print(f"  Capital final:     {r['capital_final']:.2f}")
    if sensibilidade and "sensibilidade" in r:
        print("\n  Sensibilidade a custos:")
        for s in r["sensibilidade"]:
            mult = s["multiplicador_custo"]
            print(f"    {mult}x custo -> retorno={s['retorno_liquido']:.4f}, Sharpe={s['sharpe_ratio']:.4f}, custo_R$={s['custo_total_reais']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest com custos sobre previsões CNN-LSTM (walk-forward)"
    )
    parser.add_argument("--ativo", type=str, default="VALE3", choices=["PETR4", "VALE3", "ITUB4"])
    parser.add_argument("--fold", type=int, default=1, help="Número do fold (1 a 5)")
    parser.add_argument("--estrategia", type=str, default="long_short", choices=["long_only", "long_short"])
    parser.add_argument("--sensibilidade", action="store_true", help="Rodar análise de sensibilidade a custos")
    args = parser.parse_args()

    ativo = args.ativo
    fold = args.fold
    estrategia = args.estrategia
    sensibilidade = args.sensibilidade

    print("=" * 70)
    print("BACKTEST COM CUSTOS (TCC Seção 4.5.1)")
    print("=" * 70)
    print(f"\nAtivo: {ativo} | Fold: {fold} | Estratégia: {estrategia}")

    # Dados
    from src.config import obter_nome_arquivo_dados
    arquivo = f"data/raw/{obter_nome_arquivo_dados(ativo)}"
    if not Path(arquivo).exists():
        print(f"[ERRO] Arquivo não encontrado: {arquivo}")
        return 1
    df = carregar_dados(arquivo, verbose=False)
    df_features = criar_features(df, verbose=False)
    validator = WalkForwardValidator(
        train_size=TAMANHO_TREINO_BARRAS,
        test_size=TAMANHO_TESTE_BARRAS,
        embargo=EMBARGO_BARRAS,
    )
    folds_info = validator._gerar_folds(len(df_features))
    if fold < 1 or fold > len(folds_info):
        print(f"[ERRO] Fold deve estar entre 1 e {len(folds_info)}")
        return 1
    fold_info = folds_info[fold - 1]

    print(f"\n[1/2] Dados: {len(df_features)} barras | Teste fold: [{fold_info.test_start}:{fold_info.test_end}]")
    print("[2/2] Rodando backtest...")

    try:
        resultado = _backtest_fold(
            ativo=ativo,
            fold=fold,
            df_features=df_features,
            fold_info=fold_info,
            estrategia=estrategia,
            sensibilidade=sensibilidade,
        )
    except FileNotFoundError as e:
        print(f"[ERRO] {e}")
        return 1

    # Timestamp para o nome dos arquivos (AAAAMMDD_HHMMSS)
    data_hora_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Salvar em data/backtest/ com nome detalhado
    arquivos_salvos = _salvar_resultado(resultado, data_hora_str, sensibilidade)
    if arquivos_salvos:
        print("\n[OK] Resultados salvos em:")
        for p in arquivos_salvos:
            print(f"     {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")

    print("\n" + "-" * 70)
    print("RESULTADO")
    print("-" * 70)
    _imprimir_resultado(resultado, sensibilidade=sensibilidade)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
