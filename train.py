"""
End-to-end training pipeline.

Builds features, trains all three models,
tunes ensemble weights via grid search,
and saves OOF + test predictions to disk.

Usage
-----
    python train.py               # full run
    python train.py --smoke       # quick sanity check (5000 rows, 2 folds)
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import ID_COL, TARGET_COL, PREDICTIONS_DIR, ROOT_DIR
from src.features.pipeline import build_features
from src.models.lgbm_model import train_lgbm
from src.models.xgb_model import train_xgb
from src.models.catboost_model import train_catboost
from src.utils.helpers import get_logger, timer

logger = get_logger(__name__)

SMOKE_ROWS  = 5_000
SMOKE_FOLDS = 2


def _tune_weights(
    y: np.ndarray,
    lgbm: np.ndarray,
    xgb: np.ndarray,
    catboost: np.ndarray,
    step: float = 0.02,
) -> tuple[dict[str, float], float, pd.DataFrame]:
    """Grid search over ensemble weights; returns best weights, best AUC, and full results."""
    candidates = np.arange(0, 1 + step, step)
    rows = []
    for w_lgbm in candidates:
        for w_xgb in candidates:
            w_cb = 1.0 - w_lgbm - w_xgb
            if w_cb < -1e-9:
                continue
            w_cb = max(w_cb, 0.0)
            blend = w_lgbm * lgbm + w_xgb * xgb + w_cb * catboost
            auc = roc_auc_score(y, blend)
            rows.append({"lgbm": round(w_lgbm, 4), "xgb": round(w_xgb, 4),
                         "catboost": round(w_cb, 4), "auc": round(auc, 6)})

    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    best_weights = {"lgbm": best["lgbm"], "xgb": best["xgb"], "catboost": best["catboost"]}
    return best_weights, float(best["auc"]), df


def _plot_weight_tuning(df: pd.DataFrame) -> None:
    """Scatter plot of ensemble OOF AUC across the weight grid."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df["lgbm"], df["xgb"], c=df["auc"], cmap="RdYlBu_r", alpha=0.8, s=40)
    plt.colorbar(scatter, ax=ax, label="OOF AUC")
    ax.set_xlabel("LightGBM weight")
    ax.set_ylabel("XGBoost weight")
    ax.set_title("Ensemble OOF AUC by blend weights\n(CatBoost weight = 1 - lgbm - xgb)")
    out_path = PREDICTIONS_DIR / "weight_tuning.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Weight tuning plot saved to {out_path}")


def _plot_roc_curve(y: np.ndarray, oof_blend: np.ndarray, auc: float) -> None:
    """ROC curve for ensemble OOF predictions."""
    fpr, tpr, _ = roc_curve(y, oof_blend)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=1.5, label=f"Ensemble (AUC = {auc:.5f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Ensemble OOF ROC curve")
    ax.legend(loc="lower right")
    out_path = PREDICTIONS_DIR / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help=f"Sanity-check run: {SMOKE_ROWS} rows, {SMOKE_FOLDS} folds")
    args = parser.parse_args()

    # ── Feature engineering ───────────────────────────────────────────────────
    smoke_n = SMOKE_ROWS if args.smoke else 0
    n_folds = SMOKE_FOLDS if args.smoke else None

    with timer("Feature engineering", logger):
        train, test = build_features(smoke_n=smoke_n)

    # ── Save processed datasets ───────────────────────────────────────────────
    if not args.smoke:
        processed_dir = ROOT_DIR / "processed"
        processed_dir.mkdir(exist_ok=True)
        train.to_csv(processed_dir / "train_features.csv", index=False)
        test.to_csv(processed_dir / "test_features.csv", index=False)
        logger.info(f"Processed datasets saved to {processed_dir}")

    features = [c for c in train.columns if c not in [TARGET_COL, ID_COL]]
    logger.info(f"Training with {len(features)} features")

    y = train[TARGET_COL].values

    # ── Train models ──────────────────────────────────────────────────────────
    fold_kwargs = {} if n_folds is None else {"n_folds": n_folds}

    with timer("LightGBM", logger):
        lgbm_oof, lgbm_test, _ = train_lgbm(train, test, features, **fold_kwargs)

    with timer("XGBoost", logger):
        xgb_oof, xgb_test, _ = train_xgb(train, test, features, **fold_kwargs)

    with timer("CatBoost", logger):
        cb_oof, cb_test, _ = train_catboost(train, test, features, **fold_kwargs)

    # ── Weight tuning ─────────────────────────────────────────────────────────
    with timer("Weight tuning", logger):
        best_weights, best_auc, wt_df = _tune_weights(y, lgbm_oof, xgb_oof, cb_oof)

    logger.info(f"  Best weights  lgbm={best_weights['lgbm']}  "
                f"xgb={best_weights['xgb']}  catboost={best_weights['catboost']}")
    logger.info(f"  Best ensemble OOF AUC: {best_auc:.5f}")

    oof_blend  = (best_weights["lgbm"] * lgbm_oof  + best_weights["xgb"] * xgb_oof
                  + best_weights["catboost"] * cb_oof)
    test_blend = (best_weights["lgbm"] * lgbm_test + best_weights["xgb"] * xgb_test
                  + best_weights["catboost"] * cb_test)

    PREDICTIONS_DIR.mkdir(exist_ok=True)
    _plot_weight_tuning(wt_df)
    _plot_roc_curve(y, oof_blend, best_auc)

    # ── Save OOF predictions ──────────────────────────────────────────────────
    oof_df = pd.DataFrame({
        ID_COL:            train[ID_COL].values,
        TARGET_COL:        y,
        "lgbm_oof":        lgbm_oof,
        "xgb_oof":         xgb_oof,
        "catboost_oof":    cb_oof,
        "ensemble_oof":    oof_blend,
    })
    oof_path = PREDICTIONS_DIR / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"OOF predictions saved to {oof_path}")

    # ── Save test predictions ─────────────────────────────────────────────────
    test_ids = test[ID_COL].values
    pred_path = PREDICTIONS_DIR / "test_predictions.csv"
    pd.DataFrame({ID_COL: test_ids, "TARGET": test_blend}).to_csv(pred_path, index=False)
    logger.info(f"Test predictions saved to {pred_path}")

    # ── Save results summary ──────────────────────────────────────────────────
    results = {
        "lgbm_oof_auc":     round(roc_auc_score(y, lgbm_oof), 5),
        "xgb_oof_auc":      round(roc_auc_score(y, xgb_oof), 5),
        "catboost_oof_auc": round(roc_auc_score(y, cb_oof), 5),
        "ensemble_oof_auc": round(best_auc, 5),
        "best_weights":     best_weights,
    }
    results_path = PREDICTIONS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
