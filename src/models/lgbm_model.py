import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from tqdm import tqdm

from src.config import LGBM_PARAMS, N_FOLDS, RANDOM_STATE, TARGET_COL
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def train_lgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    n_folds: int = N_FOLDS,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    X     = train[features].values
    y     = train[TARGET_COL].values
    X_test = test[features].values

    oof_preds  = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    fold_aucs  = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    for fold, (trn_idx, val_idx) in tqdm(
        enumerate(skf.split(X, y), 1), total=n_folds, desc="LightGBM folds"
    ):
        logger.info(f"Fold {fold}/{n_folds}")

        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        trn_data = lgb.Dataset(X_trn, label=y_trn)
        val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)

        params = {**LGBM_PARAMS}
        n_estimators = params.pop("n_estimators")

        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200),
        ]

        model = lgb.train(
            params,
            trn_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        oof_preds[val_idx]  = model.predict(X_val,  num_iteration=model.best_iteration)
        test_preds          += model.predict(X_test, num_iteration=model.best_iteration) / n_folds

        auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_aucs.append(auc)
        logger.info(f"  Fold {fold} AUC: {auc:.5f}")

        del model, trn_data, val_data, X_trn, y_trn, X_val, y_val
        gc.collect()

    overall_auc = roc_auc_score(y, oof_preds)
    logger.info(f"LightGBM OOF AUC: {overall_auc:.5f} | "
                f"Mean fold AUC: {np.mean(fold_aucs):.5f} ± {np.std(fold_aucs):.5f}")

    return oof_preds, test_preds, fold_aucs
