"""
Otimização da janela de look-back (n_steps) via Optuna.

Testa candidatos de janela (ex.: 16, 32, 64 barras), escolhe a melhor pela
métrica de validação (acuracia direcional) e salva resultados em diretório
separado para não misturar com o treino normal.

Uso (a partir de codigo/pipeline):
  uv run python src/scripts/otimizar_janela_optuna.py --ativo VALE3
  uv run python src/scripts/otimizar_janela_optuna.py --ativo PETR4 --n-trials 15 --folds 1,2

Resultados em: data/processed/optuna_janela/
  - best_n_steps.json
  - study_trials.csv
  - resumo_optuna_janela.txt
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Script deve ser executado a partir do diretório raiz do pipeline (codigo/pipeline)
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
import optuna
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    TAMANHO_TREINO_BARRAS,
    TAMANHO_TESTE_BARRAS,
    EMBARGO_BARRAS,
    JANELA_TEMPORAL_CANDIDATOS,
    DIR_RESULTADOS_OPTUNA_JANELA,
    SEED,
    HIPERPARAMETROS_PADRAO_CNN_LSTM,
)
from src.data_processing.load_data import carregar_dados
from src.data_processing.feature_engineering import criar_features
from src.data_processing.prepare_sequences import preparar_dados_dl
from src.utils.validation import WalkForwardValidator
from src.models.cnn_lstm_model import criar_modelo_cnn_lstm

np.random.seed(SEED)
tf.random.set_seed(SEED)


def _avaliar_janela(
    n_steps: int,
    df_features: pd.DataFrame,
    fold,
    epochs: int,
    verbose: int,
) -> float:
    """
    Para um n_steps fixo, monta dados do fold, treina e retorna
    acurácia direcional na validação (sem usar teste).
    """
    X_train, y_train, _, _, _, _ = preparar_dados_dl(
        df_features,
        fold.train_start,
        fold.train_end,
        fold.test_start,
        fold.test_end,
        n_steps=n_steps,
        verbose=False,
    )
    n_features = X_train.shape[2]

    # Split treino / validação interna (80/20), temporal
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    # Remover neutros (igual ao optuna_optimizer)
    mask_tr = y_tr != 0
    mask_val = y_val != 0
    X_tr_f = X_tr[mask_tr]
    y_tr_f = y_tr[mask_tr]
    X_val_f = X_val[mask_val]
    y_val_f = y_val[mask_val]
    y_tr_bin = np.where(y_tr_f == 1, 1, 0)
    y_val_bin = np.where(y_val_f == 1, 1, 0)

    if len(X_tr_f) < 10 or len(X_val_f) < 5:
        return 0.0

    # Class weights
    if len(np.unique(y_tr_bin)) > 1:
        classes = np.unique(y_tr_bin)
        weights = compute_class_weight("balanced", classes=classes, y=y_tr_bin)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    else:
        class_weight = None

    defaults = HIPERPARAMETROS_PADRAO_CNN_LSTM
    model = criar_modelo_cnn_lstm(
        n_steps=n_steps,
        n_features=n_features,
        conv_filters=defaults["conv_filters"],
        conv_kernel_size=defaults["conv_kernel_size"],
        pool_size=defaults["pool_size"],
        lstm_units=defaults["lstm_units"],
        dropout=defaults["dropout"],
        learning_rate=defaults["learning_rate"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        ),
    ]

    try:
        keras.backend.clear_session()
        model.fit(
            X_tr_f,
            y_tr_bin,
            validation_data=(X_val_f, y_val_bin),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=verbose,
        )
        y_pred_proba = model.predict(X_val_f, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_pred_dir = np.where(y_pred == 1, 1, -1)
        acuracia = np.mean(y_pred_dir == y_val_f)
        return float(acuracia)
    except Exception as e:
        if verbose:
            print(f"     n_steps={n_steps} erro: {e}")
        keras.backend.clear_session()
        return 0.0


def main(
    ativo: str = "VALE3",
    arquivo_dados: str | None = None,
    n_trials: int = 12,
    epochs: int = 50,
    folds_para_objetivo: str = "1",
    verbose: bool = True,
) -> None:
    """
    Roda estudo Optuna para escolher a melhor janela de look-back.
    Resultados salvos em data/processed/optuna_janela/ (separados do treino normal).
    """
    out_dir = Path("data/processed") / DIR_RESULTADOS_OPTUNA_JANELA
    out_dir.mkdir(parents=True, exist_ok=True)

    if arquivo_dados is None:
        from src.config import obter_nome_arquivo_dados
        arquivo_dados = f"data/raw/{obter_nome_arquivo_dados(ativo)}"
    if not Path(arquivo_dados).exists():
        print(f"[ERRO] Arquivo não encontrado: {arquivo_dados}")
        return

    print("=" * 70)
    print("OTIMIZAÇÃO DE JANELA (LOOK-BACK) VIA OPTUNA")
    print("=" * 70)
    print(f"Ativo: {ativo}  |  Candidatos: {JANELA_TEMPORAL_CANDIDATOS}")
    print(f"Resultados em: {out_dir.resolve()}")
    print()

    df = carregar_dados(arquivo_dados, verbose=False)
    df_features = criar_features(df, verbose=False)

    validator = WalkForwardValidator(
        train_size=TAMANHO_TREINO_BARRAS,
        test_size=TAMANHO_TESTE_BARRAS,
        embargo=EMBARGO_BARRAS,
    )
    folds = validator._gerar_folds(len(df_features))

    # Folds usados na função objetivo (ex.: 1 ou 1,2)
    try:
        indices = [int(x.strip()) for x in folds_para_objetivo.split(",")]
        indices = [i for i in indices if 1 <= i <= len(folds)]
        if not indices:
            indices = [1]
        folds_use = [folds[i - 1] for i in indices]
    except ValueError:
        folds_use = [folds[0]]

    def objetivo(trial: optuna.Trial) -> float:
        n_steps = trial.suggest_categorical("n_steps", JANELA_TEMPORAL_CANDIDATOS)
        # Média da acurácia de validação em um ou mais folds
        valores = []
        for fold in folds_use:
            v = _avaliar_janela(n_steps, df_features, fold, epochs=epochs, verbose=0)
            valores.append(v)
        return float(np.mean(valores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name=f"optuna_janela_{ativo}",
    )
    study.optimize(objetivo, n_trials=n_trials, show_progress_bar=verbose)

    best_n_steps = int(study.best_params["n_steps"])
    best_value = study.best_value

    # Salvar resultados (separados do treino normal)
    best_path = out_dir / "best_n_steps.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_n_steps": best_n_steps,
                "best_accuracy_direcional_val": best_value,
                "ativo": ativo,
                "candidatos": JANELA_TEMPORAL_CANDIDATOS,
                "n_trials": n_trials,
            },
            f,
            indent=2,
        )
    print(f"[OK] Melhor n_steps salvo em: {best_path}")

    trials_path = out_dir / "study_trials.csv"
    df_trials = study.trials_dataframe()
    df_trials.to_csv(trials_path, index=False)
    print(f"[OK] Trials salvos em: {trials_path}")

    resumo_path = out_dir / "resumo_optuna_janela.txt"
    with open(resumo_path, "w", encoding="utf-8") as f:
        f.write(f"Otimização de janela (Optuna) - {ativo}\n")
        f.write(f"Candidatos: {JANELA_TEMPORAL_CANDIDATOS}\n")
        f.write(f"Melhor n_steps: {best_n_steps}\n")
        f.write(f"Acurácia direcional (validação): {best_value:.4f}\n")
        f.write(f"Trials: {n_trials}\n")
    print(f"[OK] Resumo em: {resumo_path}")

    print()
    print("=" * 70)
    print(f"Melhor janela: n_steps = {best_n_steps}  (acuracia val = {best_value:.4f})")
    print("=" * 70)
    print("Para treinar com essa janela, use JANELA_TEMPORAL_STEPS no config ou")
    print("um script que leia best_n_steps.json e passe n_steps explicitamente.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Otimizar janela de look-back via Optuna")
    p.add_argument("--ativo", type=str, default="VALE3")
    p.add_argument("--arquivo", type=str, default=None)
    p.add_argument("--n-trials", type=int, default=12)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--folds", type=str, default="1", help="Folds para objetivo, ex: 1 ou 1,2")
    p.add_argument("--quiet", action="store_true", help="Menos impressão")
    args = p.parse_args()
    main(
        ativo=args.ativo,
        arquivo_dados=args.arquivo,
        n_trials=args.n_trials,
        epochs=args.epochs,
        folds_para_objetivo=args.folds,
        verbose=not args.quiet,
    )
