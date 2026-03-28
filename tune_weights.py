"""
Grid search over model weights to find the combination that maximises ensemble OOF AUC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from src.config import PREDICTIONS_DIR, TARGET_COL, ID_COL

def load_oof() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load OOF predictions for all three models and true labels."""
    oof_path = PREDICTIONS_DIR / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF file not found: {oof_path}. Run train.py first.")
    oof_df = pd.read_csv(oof_path)
    y        = oof_df[TARGET_COL].values
    lgbm     = oof_df["lgbm_oof"].values
    xgb      = oof_df["xgb_oof"].values
    catboost = oof_df["catboost_oof"].values
    return y, lgbm, xgb, catboost


def grid_search(step: float = 0.02) -> pd.DataFrame:
    """Try all weight combinations on a grid and return results sorted by AUC."""
    y, lgbm, xgb, catboost = load_oof()
    weights = np.arange(0, 1 + step, step)
    results = []

    for w_lgbm in weights:
        for w_xgb in weights:
            w_cb = 1.0 - w_lgbm - w_xgb
            if w_cb < -1e-9:
                continue
            w_cb = max(w_cb, 0.0)
            blend = w_lgbm * lgbm + w_xgb * xgb + w_cb * catboost
            auc = roc_auc_score(y, blend)
            results.append({"lgbm": round(w_lgbm, 4), "xgb": round(w_xgb, 4),
                            "catboost": round(w_cb, 4), "auc": round(auc, 6)})

    return pd.DataFrame(results).sort_values("auc", ascending=False).reset_index(drop=True)


def plot_results(df: pd.DataFrame) -> None:
    """Plot ensemble OOF AUC as a heatmap: LightGBM weight on x-axis, XGBoost weight on y-axis"""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df["lgbm"], df["xgb"], c=df["auc"], cmap="RdYlBu_r",
                         alpha=0.8, s=40)
    plt.colorbar(scatter, ax=ax, label="OOF AUC")
    ax.set_xlabel("LightGBM weight")
    ax.set_ylabel("XGBoost weight")
    ax.set_title("Ensemble OOF AUC by blend weights\n(CatBoost weight = 1 - lgbm - xgb)")
    out_path = PREDICTIONS_DIR / "weight_tuning.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to {out_path}")


def main() -> None:
    df = grid_search()

    print("\nTop 10 weight combinations:")
    print(df.head(10).to_string(index=False))

    best = df.iloc[0]
    print(f"\nBest weights: lgbm={best['lgbm']}, xgb={best['xgb']}, catboost={best['catboost']}")
    print(f"Best OOF AUC: {best['auc']:.5f}")

    plot_results(df)

    model_test_paths = {k: PREDICTIONS_DIR / f"{k}_test.csv" for k in ["lgbm", "xgb", "catboost"]}
    missing = [str(p) for p in model_test_paths.values() if not p.exists()]
    if missing:
        print(f"Cannot generate test predictions — missing files: {missing}")
    else:
        test_ids = pd.read_csv(PREDICTIONS_DIR / "test_predictions.csv")[ID_COL]
        test_blend = (
            best["lgbm"]     * pd.read_csv(model_test_paths["lgbm"])["TARGET"].values +
            best["xgb"]      * pd.read_csv(model_test_paths["xgb"])["TARGET"].values +
            best["catboost"] * pd.read_csv(model_test_paths["catboost"])["TARGET"].values
        )
        out_path = PREDICTIONS_DIR / "test_predictions_tuned.csv"
        pd.DataFrame({ID_COL: test_ids, "TARGET": test_blend}).to_csv(out_path, index=False)
        print(f"Tuned test predictions saved to {out_path}")


if __name__ == "__main__":
    main()
