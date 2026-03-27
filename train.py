import argparse
import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import ID_COL, TARGET_COL, PREDICTIONS_DIR, ROOT_DIR
from src.features.pipeline import build_features
from src.models.lgbm_model import train_lgbm
from src.models.xgb_model import train_xgb
from src.models.catboost_model import train_catboost
from src.models.ensemble import blend
from src.utils.helpers import get_logger, timer

logger = get_logger(__name__)

SMOKE_ROWS  = 5_000
SMOKE_FOLDS = 2

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
        lgbm_oof, lgbm_test, lgbm_aucs = train_lgbm(train, test, features, **fold_kwargs)

    with timer("XGBoost", logger):
        xgb_oof, xgb_test, xgb_aucs = train_xgb(train, test, features, **fold_kwargs)

    with timer("CatBoost", logger):
        cb_oof, cb_test, cb_aucs = train_catboost(train, test, features, **fold_kwargs)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    with timer("Ensemble", logger):
        oof_blend, test_blend = blend(
            oof_dict  = {"lgbm": lgbm_oof,  "xgb": xgb_oof,  "catboost": cb_oof},
            test_dict = {"lgbm": lgbm_test, "xgb": xgb_test, "catboost": cb_test},
            y         = y,
        )

    # ── Save OOF predictions ──────────────────────────────────────────────────
    oof_df = pd.DataFrame({
        ID_COL:            train[ID_COL].values,
        TARGET_COL:        y,
        "lgbm_oof":        lgbm_oof,
        "xgb_oof":         xgb_oof,
        "catboost_oof":    cb_oof,
        "ensemble_oof":    oof_blend,
    })
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    oof_path = PREDICTIONS_DIR / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"OOF predictions saved to {oof_path}")

    # ── Save per-model test predictions (for reblending) ─────────────────────
    test_ids = test[ID_COL].values
    for name, preds in [("lgbm", lgbm_test), ("xgb", xgb_test), ("catboost", cb_test)]:
        p = PREDICTIONS_DIR / f"{name}_test.csv"
        pd.DataFrame({ID_COL: test_ids, "TARGET": preds}).to_csv(p, index=False)
        logger.info(f"{name} test predictions saved to {p}")

    # ── Save ensemble test predictions ────────────────────────────────────────
    pred_path = PREDICTIONS_DIR / "test_predictions.csv"
    pd.DataFrame({ID_COL: test_ids, "TARGET": test_blend}).to_csv(pred_path, index=False)
    logger.info(f"Ensemble test predictions saved to {pred_path}")

    # ── Save results summary ──────────────────────────────────────────────────
    results = {
        "lgbm_oof_auc":     round(roc_auc_score(y, lgbm_oof), 5),
        "xgb_oof_auc":      round(roc_auc_score(y, xgb_oof), 5),
        "catboost_oof_auc": round(roc_auc_score(y, cb_oof), 5),
        "ensemble_oof_auc": round(roc_auc_score(y, oof_blend), 5),
    }
    results_path = PREDICTIONS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()