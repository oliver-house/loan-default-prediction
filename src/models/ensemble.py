"""
Weighted ensemble of LightGBM, XGBoost, and CatBoost OOF predictions
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import ENSEMBLE_WEIGHTS
from src.utils.helpers import get_logger

logger = get_logger(__name__)

def blend(
    oof_dict: dict[str, np.ndarray],
    test_dict: dict[str, np.ndarray],
    y: np.ndarray,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted average of OOF and test predictions from multiple models
    """
    if weights is None:
        weights = ENSEMBLE_WEIGHTS

    total = sum(weights[k] for k in oof_dict)
    norm  = {k: weights[k] / total for k in oof_dict}

    oof_blend  = sum(norm[k] * oof_dict[k]  for k in oof_dict)
    test_blend = sum(norm[k] * test_dict[k] for k in test_dict)

    for name, preds in oof_dict.items():
        auc = roc_auc_score(y, preds)
        logger.info(f"  {name:12s} OOF AUC: {auc:.5f}  (weight={norm[name]:.2f})")

    ensemble_auc = roc_auc_score(y, oof_blend)
    logger.info(f"  {'Ensemble':12s} OOF AUC: {ensemble_auc:.5f}")

    return oof_blend, test_blend
