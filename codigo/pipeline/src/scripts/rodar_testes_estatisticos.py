"""Script para rodar testes estatísticos de comparação CNN-LSTM vs baselines.

Conforme TCC Seção 4.5.2: comparação da série de perdas/erros do modelo proposto
contra baselines por meio do teste de Diebold-Mariano (DIEBOLD; MARIANO, 1995),
avaliando significância das diferenças de acurácia direcional.

Uso:
  uv run python src/scripts/rodar_testes_estatisticos.py --ativo PETR4
  uv run python src/scripts/rodar_testes_estatisticos.py --ativo VALE3
  uv run python src/scripts/rodar_testes_estatisticos.py --ativo ITUB4
  uv run python src/scripts/rodar_testes_estatisticos.py --todos
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Suprimir avisos do statsmodels (frequência do índice, convergência ARIMA)
# O script roda 5 folds × 4 baselines; ARIMA testa várias (p,d,q) por fold → muitas mensagens
warnings.filterwarnings("ignore", module="statsmodels")

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

from src.data_processing.load_data import carregar_dados
from src.data_processing.feature_engineering import criar_features
from src.data_processing.prepare_sequences import preparar_dados_dl
from src.models.baselines import NaiveBaseline, DriftBaseline, ARIMABaseline
from src.models.prophet_model import ProphetBaseline
from src.utils.validation import WalkForwardValidator
from src.utils.metrics import calcular_acuracia_direcional
from src.utils.diebold_mariano import (
    perda_direcional,
    perda_brier,
    resumo_dm,
    segmentar_por_volatilidade,
)
from src.config import (
    TAMANHO_TREINO_BARRAS,
    TAMANHO_TESTE_BARRAS,
    EMBARGO_BARRAS,
    obter_nome_arquivo_dados,
    JANELA_TEMPORAL_STEPS,
)
from src.utils.focal_loss import focal_loss

try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False


def carregar_modelo_fold(ativo: str, modelo_tipo: str, fold: int) -> "keras.Model":
    """Carrega modelo salvo de um fold (CNN-LSTM com focal loss)."""
    model_path = Path("models") / ativo / modelo_tipo / f"fold_{fold}_checkpoint.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    custom_objects = {"focal_loss_fixed": focal_loss(gamma=5.0, alpha=0.5)}
    return keras.models.load_model(model_path, custom_objects=custom_objects)


def obter_previsoes_cnn_lstm(
    ativo: str,
    df_features: pd.DataFrame,
    folds_info: list,
    modelo_tipo: str = "cnn_lstm",
    retornar_proba: bool = False,
) -> tuple:
    """Obtém y_true e y_pred (e opcionalmente y_proba) do CNN-LSTM para todos os folds."""
    list_y_true = []
    list_y_pred = []
    list_y_proba = [] if retornar_proba else None
    for i, fold in enumerate(folds_info):
        try:
            model = carregar_modelo_fold(ativo, modelo_tipo, i + 1)
        except FileNotFoundError:
            continue
        X_train, y_train, X_test, y_test, scaler, _ = preparar_dados_dl(
            df_features,
            fold.train_start, fold.train_end,
            fold.test_start, fold.test_end,
            n_steps=JANELA_TEMPORAL_STEPS,
            verbose=False,
        )
        y_pred_proba = model.predict(X_test, verbose=0)
        proba_flat = y_pred_proba.flatten()
        y_pred = np.where(proba_flat > 0.5, 1, -1)
        list_y_true.append(y_test)
        list_y_pred.append(y_pred)
        if retornar_proba:
            list_y_proba.append(proba_flat)
    if not list_y_true:
        if retornar_proba:
            return np.array([]), np.array([]), np.array([])
        return np.array([]), np.array([])
    y_true = np.concatenate(list_y_true)
    y_pred = np.concatenate(list_y_pred)
    if retornar_proba:
        y_proba = np.concatenate(list_y_proba)
        return y_true, y_pred, y_proba
    return y_true, y_pred


def obter_volatilidade_cnn(
    df_features: pd.DataFrame,
    folds_info: list,
    n_steps: int,
) -> np.ndarray:
    """Volatilidade alinhada às barras de teste do CNN (test_start+n_steps até test_end por fold)."""
    if "volatility" not in df_features.columns:
        return np.array([])
    parts = []
    for fold in folds_info:
        part = df_features["volatility"].iloc[
            fold.test_start + n_steps : fold.test_end
        ].values
        parts.append(part)
    return np.concatenate(parts) if parts else np.array([])


def obter_y_true_global(df_features: pd.DataFrame, folds_info: list) -> np.ndarray:
    """Constrói série de target (direção) para todos os testes, na ordem dos folds."""
    parts = []
    for fold in folds_info:
        part = df_features["target"].iloc[fold.test_start:fold.test_end].values
        parts.append(part)
    return np.concatenate(parts) if parts else np.array([])


def trim_baseline_pred_to_cnn(all_y_pred_baseline: np.ndarray, folds_info: list, n_steps: int) -> np.ndarray:
    """Corta previsões do baseline para alinhar com CNN (mesmo número de pontos por fold).

    O CNN prevê apenas a partir da barra test_start + n_steps (janela). O baseline prevê
    desde test_start. Para comparar no mesmo conjunto, descartamos as primeiras n_steps
    previsões do baseline em cada fold.
    """
    parts = []
    start = 0
    for fold in folds_info:
        fold_len = fold.test_end - fold.test_start
        # Manter apenas previsões que correspondem às mesmas barras que o CNN
        trim = all_y_pred_baseline[start + n_steps : start + fold_len]
        parts.append(trim)
        start += fold_len
    return np.concatenate(parts) if parts else np.array([])


def _row_dm(ativo, baseline, perda_cnn, perda_base, diff, dm_stat, dm_pval, n_obs, regime="geral"):
    """Uma linha do DataFrame de resultados DM (com coluna Regime)."""
    return {
        "Ativo": ativo,
        "Baseline": baseline,
        "Regime": regime,
        "N_obs": n_obs,
        "Perda_media_CNN": perda_cnn,
        "Perda_media_Baseline": perda_base,
        "Diferenca_perda": diff,
        "DM_statistic": dm_stat,
        "DM_pvalue": dm_pval,
    }


def obter_previsoes_baseline(
    baseline,
    df_features: pd.DataFrame,
    validator: WalkForwardValidator,
) -> np.ndarray:
    """Obtém all_y_pred de um baseline via walk-forward (ordem igual a y_true_global)."""
    results = validator.validate(
        model=baseline,
        X=df_features,
        y=df_features["returns"],
        fit_func=lambda m, X, y: m.fit(y),
        predict_func=lambda m, X: m.predict(steps=len(X)) if hasattr(m, "predict") else m.predict(),
        verbose=False,
    )
    return results["all_y_pred"]


def rodar_testes_ativo(
    ativo: str,
    df_features: pd.DataFrame,
    validator: WalkForwardValidator,
    baselines_config: list,
    com_regimes: bool = False,
    com_brier: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Roda testes de Diebold-Mariano para um ativo: CNN-LSTM vs cada baseline.

    com_regimes: se True, segmenta por volatilidade (baixa/alta) e adiciona linhas com Regime.
    com_brier: se True, inclui teste DM sobre perda Brier (além da direcional).

    Retorna:
        df_dm: uma linha por (ativo, baseline[, Regime]) com estatística DM e p-valor.
        df_folds: acurácias por fold para teste pareado (opcional).
    """
    folds_info = validator.folds
    if not folds_info:
        return pd.DataFrame(), pd.DataFrame()

    n_steps = JANELA_TEMPORAL_STEPS
    # Previsões CNN-LSTM (y_true_cnn, y_pred_cnn e opcionalmente y_proba)
    if not KERAS_AVAILABLE:
        print("[AVISO] TensorFlow não disponível; pulando CNN-LSTM.")
        y_true_cnn = np.array([])
        y_pred_cnn = np.array([])
        y_proba_cnn = np.array([])
    else:
        out = obter_previsoes_cnn_lstm(
            ativo, df_features, folds_info, retornar_proba=com_brier
        )
        if len(out) == 3:
            y_true_cnn, y_pred_cnn, y_proba_cnn = out
        else:
            y_true_cnn, y_pred_cnn = out
            y_proba_cnn = np.array([])

    if len(y_true_cnn) == 0:
        return pd.DataFrame(), pd.DataFrame()

    loss_cnn = perda_direcional(y_true_cnn, y_pred_cnn)
    n_obs = len(loss_cnn)
    # Volatilidade alinhada às barras do CNN (para segmentação por regime)
    vol_cnn = (
        obter_volatilidade_cnn(df_features, folds_info, n_steps)
        if com_regimes and "volatility" in df_features.columns
        else np.array([])
    )
    if com_regimes and len(vol_cnn) != len(y_true_cnn):
        vol_cnn = np.array([])

    rows_dm = []
    acc_cnn_por_fold = []
    start = 0
    for fold in folds_info:
        fold_len_full = fold.test_end - fold.test_start
        fold_len_cnn = fold_len_full - n_steps
        if fold_len_cnn > 0:
            y_true_f = y_true_cnn[start : start + fold_len_cnn]
            y_pred_cnn_f = y_pred_cnn[start : start + fold_len_cnn]
            if (y_true_f != 0).any():
                acc_cnn_por_fold.append(calcular_acuracia_direcional(y_true_f, y_pred_cnn_f))
            else:
                acc_cnn_por_fold.append(np.nan)
            start += fold_len_cnn

    for nome_baseline, baseline_instance in baselines_config:
        try:
            all_y_pred_b = obter_previsoes_baseline(baseline_instance, df_features, validator)
            y_pred_b = trim_baseline_pred_to_cnn(all_y_pred_b, folds_info, n_steps)
        except Exception as e:
            print(f"[AVISO] Baseline {nome_baseline} falhou: {e}")
            rows_dm.append(_row_dm(ativo, nome_baseline, np.mean(loss_cnn), np.nan, np.nan, np.nan, np.nan, n_obs, "geral"))
            continue

        # Alinhar tamanho (baseline já cortado para mesmo conjunto que CNN)
        min_len = min(len(y_true_cnn), len(y_pred_b))
        y_true_a = y_true_cnn[:min_len]
        y_pred_cnn_a = y_pred_cnn[:min_len]
        y_pred_b_a = y_pred_b[:min_len]

        loss_cnn_a = perda_direcional(y_true_a, y_pred_cnn_a)
        loss_b_a = perda_direcional(y_true_a, y_pred_b_a)

        if len(loss_cnn_a) < 10:
            rows_dm.append(_row_dm(ativo, nome_baseline, np.mean(loss_cnn_a), np.mean(loss_b_a), np.nan, np.nan, np.nan, len(loss_cnn_a), "geral"))
            continue

        res = resumo_dm(
            loss_cnn_a, loss_b_a,
            nome_a="CNN-LSTM", nome_b=nome_baseline,
            h=1,
        )
        rows_dm.append(_row_dm(
            ativo, nome_baseline,
            res["perda_media_a"], res["perda_media_b"],
            res["diferenca_perda"], res["dm_statistic"], res["dm_pvalue"],
            res["n_obs"], "geral",
        ))

        # Segmentação por regime de volatilidade (TCC 4.5.2)
        if com_regimes and len(vol_cnn) >= len(y_true_a):
            mask_nz = y_true_a != 0
            vol_nz = vol_cnn[: len(y_true_a)][mask_nz]
            if len(vol_nz) == len(loss_cnn_a) and len(vol_nz) >= 20:
                (loss_cnn_baixa, loss_b_baixa), (loss_cnn_alta, loss_b_alta) = segmentar_por_volatilidade(
                    vol_nz, loss_cnn_a, loss_b_a, percentil=50.0
                )
                for regime, la, lb in [
                    ("baixa_vol", loss_cnn_baixa, loss_b_baixa),
                    ("alta_vol", loss_cnn_alta, loss_b_alta),
                ]:
                    if len(la) >= 10:
                        r = resumo_dm(la, lb, nome_a="CNN-LSTM", nome_b=nome_baseline, h=1)
                        rows_dm.append(_row_dm(
                            ativo, nome_baseline,
                            r["perda_media_a"], r["perda_media_b"],
                            r["diferenca_perda"], r["dm_statistic"], r["dm_pvalue"],
                            r["n_obs"], regime,
                        ))

        # DM sobre perda Brier (TCC 4.5.2)
        if com_brier and len(y_proba_cnn) >= len(y_true_a):
            proba_a = y_proba_cnn[: len(y_true_a)]
            proba_b = (y_pred_b_a > 0).astype(float)
            loss_brier_cnn = perda_brier(y_true_a, proba_a)
            loss_brier_b = perda_brier(y_true_a, proba_b)
            if len(loss_brier_cnn) >= 10:
                res_b = resumo_dm(
                    loss_brier_cnn, loss_brier_b,
                    nome_a="CNN-LSTM", nome_b=nome_baseline, h=1,
                )
                rows_dm.append(_row_dm(
                    ativo, nome_baseline,
                    res_b["perda_media_a"], res_b["perda_media_b"],
                    res_b["diferenca_perda"], res_b["dm_statistic"], res_b["dm_pvalue"],
                    res_b["n_obs"], "brier",
                ))

    df_dm = pd.DataFrame(rows_dm)

    # Opcional: teste pareado por folds (se tivéssemos acurácia por fold dos baselines)
    df_folds = pd.DataFrame()
    return df_dm, df_folds


def main():
    parser = argparse.ArgumentParser(
        description="Testes estatísticos (Diebold-Mariano) CNN-LSTM vs baselines"
    )
    parser.add_argument(
        "--ativo",
        type=str,
        default=None,
        help="Ativo (PETR4, VALE3, ITUB4). Use --todos para rodar os 3.",
    )
    parser.add_argument(
        "--todos",
        action="store_true",
        help="Rodar para PETR4, VALE3 e ITUB4.",
    )
    parser.add_argument(
        "--saida",
        type=str,
        default="data/processed/testes_diebold_mariano.csv",
        help="Caminho do CSV de saída.",
    )
    parser.add_argument(
        "--regimes",
        action="store_true",
        help="Segmentar por regime de volatilidade (baixa/alta) e incluir DM por regime.",
    )
    parser.add_argument(
        "--brier",
        action="store_true",
        help="Incluir teste DM sobre perda Brier (além da acurácia direcional).",
    )
    args = parser.parse_args()

    if args.todos:
        ativos = ["PETR4", "VALE3", "ITUB4"]
    elif args.ativo:
        ativos = [args.ativo]
    else:
        parser.error("Informe --ativo OU --todos")
        return

    baselines_config = [
        ("Naive", NaiveBaseline()),
        ("Drift", DriftBaseline()),
        ("ARIMA", ARIMABaseline()),
        ("Prophet", ProphetBaseline()),
    ]

    print("TESTES ESTATÍSTICOS - DIEBOLD-MARIANO")
    print("CNN-LSTM vs Baselines (acurácia direcional)")
    print("=" * 60)

    all_dm = []
    for ativo in ativos:
        arquivo = f"data/raw/{obter_nome_arquivo_dados(ativo)}"
        if not os.path.exists(arquivo):
            print(f"[AVISO] Dados não encontrados: {arquivo}")
            continue
        print(f"\n[1/3] Carregando dados: {ativo}...")
        df = carregar_dados(arquivo, verbose=False)
        df_features = criar_features(df, verbose=False)
        print(f"      {df_features.shape[0]} barras")

        print(f"[2/3] Walk-forward e previsões CNN-LSTM...")
        validator = WalkForwardValidator(
            train_size=TAMANHO_TREINO_BARRAS,
            test_size=TAMANHO_TESTE_BARRAS,
            embargo=EMBARGO_BARRAS,
        )
        validator.folds = validator._gerar_folds(len(df_features))
        print(f"      {len(validator.folds)} folds")

        print(f"[3/3] Diebold-Mariano vs baselines..." + (" (regimes)" if args.regimes else "") + (" + Brier" if args.brier else ""))
        df_dm, _ = rodar_testes_ativo(
            ativo, df_features, validator, baselines_config,
            com_regimes=args.regimes,
            com_brier=args.brier,
        )
        if not df_dm.empty:
            all_dm.append(df_dm)
            cols = ["Baseline", "Regime", "N_obs", "Perda_media_CNN", "Perda_media_Baseline", "DM_statistic", "DM_pvalue"]
            print(df_dm[[c for c in cols if c in df_dm.columns]].to_string(index=False))

    if not all_dm:
        print("\n[ERRO] Nenhum resultado gerado.")
        return

    df_final = pd.concat(all_dm, ignore_index=True)
    out_path = Path(args.saida)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_path, index=False)
    print(f"\n[OK] Resultados salvos em: {out_path}")
    print("\nInterpretação: DM_pvalue < 0.05 indica diferença significativa entre CNN-LSTM e o baseline (rejeita H0: mesma acurácia).")


if __name__ == "__main__":
    main()
