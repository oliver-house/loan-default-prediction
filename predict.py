import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import PREDICTIONS_DIR, ID_COL, TARGET_COL, ENSEMBLE_WEIGHTS
from src.utils.helpers import get_logger

logger = get_logger(__name__)

def reblend(weights: dict[str, float]) -> None:
    """Re-blend saved OOF predictions with custom weights and save new test predictions."""
    oof_path = PREDICTIONS_DIR / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF file not found: {oof_path}. Run train.py first.")

    oof_df = pd.read_csv(oof_path)
    y      = oof_df[TARGET_COL].values

    model_cols = {"lgbm": "lgbm_oof", "xgb": "xgb_oof", "catboost": "catboost_oof"}
    total      = sum(weights[k] for k in model_cols)
    norm       = {k: weights[k] / total for k in model_cols}

    oof_blend = sum(norm[k] * oof_df[col].values for k, col in model_cols.items())
    logger.info(f"Re-blended OOF AUC: {roc_auc_score(y, oof_blend):.5f}")
    logger.info(f"Weights used: { {k: f'{v:.2f}' for k, v in norm.items()} }")

    # Apply the same weights to test predictions
    test_path = PREDICTIONS_DIR / "test_predictions.csv"
    if not test_path.exists():
        logger.warning("test_predictions.csv not found — cannot re-blend test set.")
        return

    # We need individual model test predictions to reblend; read from per-model files if present
    model_test_paths = {k: PREDICTIONS_DIR / f"{k}_test.csv" for k in model_cols}
    missing = [str(p) for p in model_test_paths.values() if not p.exists()]
    if missing:
        logger.warning(
            "Per-model test prediction files not found; cannot re-blend test set.\n"
            f"Missing: {missing}"
        )
        return

    test_ids   = pd.read_csv(test_path)[ID_COL]
    test_blend = sum(
        norm[k] * pd.read_csv(model_test_paths[k])["TARGET"].values
        for k in model_cols
    )

    out = pd.DataFrame({ID_COL: test_ids, "TARGET": test_blend})
    out_path = PREDICTIONS_DIR / "test_predictions_reblended.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Re-blended test predictions saved to {out_path}")


def summarise() -> None:
    """Print a summary of saved OOF prediction quality."""
    oof_path = PREDICTIONS_DIR / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF file not found: {oof_path}. Run train.py first.")

    oof_df = pd.read_csv(oof_path)
    y      = oof_df[TARGET_COL].values

    cols = {
        "LightGBM":  "lgbm_oof",
        "XGBoost":   "xgb_oof",
        "CatBoost":  "catboost_oof",
        "Ensemble":  "ensemble_oof",
    }
    print("\n── OOF AUC Summary ─────────────────────")
    for name, col in cols.items():
        if col in oof_df.columns:
            auc = roc_auc_score(y, oof_df[col].values)
            print(f"  {name:12s}: {auc:.5f}")
    print("────────────────────────────────────────\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict / summarise credit risk model outputs.")
    parser.add_argument("--summarise", action="store_true", help="Print OOF AUC summary.")
    parser.add_argument("--reblend",   action="store_true", help="Re-blend with custom weights.")
    parser.add_argument("--lgbm",      type=float, default=ENSEMBLE_WEIGHTS["lgbm"],
                        help="LightGBM blend weight")
    parser.add_argument("--xgb",       type=float, default=ENSEMBLE_WEIGHTS["xgb"],
                        help="XGBoost blend weight")
    parser.add_argument("--catboost",  type=float, default=ENSEMBLE_WEIGHTS["catboost"],
                        help="CatBoost blend weight")
    args = parser.parse_args()

    if args.summarise:
        summarise()
    elif args.reblend:
        reblend({"lgbm": args.lgbm, "xgb": args.xgb, "catboost": args.catboost})
    else:
        summarise()


if __name__ == "__main__":
    main()
