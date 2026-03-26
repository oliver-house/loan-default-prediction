import gc
import pandas as pd

from src.config import DATA_FILES, RANDOM_STATE
from src.utils.helpers import get_logger, timer
from src.features.application import process_application
from src.features.bureau import process_bureau
from src.features.previous_application import process_previous_application

logger = get_logger(__name__)

def build_features(smoke_n: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    # ── Application (main table) ──────────────────────────────────────────────
    with timer("Application features", logger):
        train = pd.read_csv(DATA_FILES["train"])
        test  = pd.read_csv(DATA_FILES["test"])
        logger.info(f"Train: {train.shape}, Test: {test.shape}")

        if smoke_n:
            train = train.sample(
                n=min(smoke_n, len(train)), random_state=RANDOM_STATE
            ).reset_index(drop=True)
            logger.info(f"[SMOKE] Sampled train to {len(train)} rows")

        smoke_ids: set | None = set(train["SK_ID_CURR"]) if smoke_n else None

        train = process_application(train)
        test  = process_application(test)

        # Align columns after OHE (train may have categories absent in test)
        train, test = train.align(test, join="left", axis=1)
        test = test.fillna(0)

    # ── Bureau ────────────────────────────────────────────────────────────────
    with timer("Bureau features", logger):
        bureau         = pd.read_csv(DATA_FILES["bureau"])
        bureau_balance = pd.read_csv(DATA_FILES["bureau_balance"])
        if smoke_ids:
            bureau = bureau[bureau["SK_ID_CURR"].isin(smoke_ids)].reset_index(drop=True)
            bureau_balance = bureau_balance[
                bureau_balance["SK_ID_BUREAU"].isin(bureau["SK_ID_BUREAU"])
            ].reset_index(drop=True)
        bureau_agg = process_bureau(bureau, bureau_balance)
        del bureau, bureau_balance; gc.collect()

        train = train.merge(bureau_agg, on="SK_ID_CURR", how="left")
        test  = test.merge(bureau_agg,  on="SK_ID_CURR", how="left")
        del bureau_agg; gc.collect()

    # ── Previous applications ────────────────────────────────────────────────
    with timer("Previous application features", logger):
        prev = pd.read_csv(DATA_FILES["prev_app"])
        if smoke_ids:
            prev = prev[prev["SK_ID_CURR"].isin(smoke_ids)].reset_index(drop=True)
        prev_agg = process_previous_application(prev)
        del prev; gc.collect()
        train = train.merge(prev_agg, on="SK_ID_CURR", how="left")
        test  = test.merge(prev_agg,  on="SK_ID_CURR", how="left")
        del prev_agg; gc.collect()

    # ── POS CASH balance ──────────────────────────────────────────────────────
    with timer("POS CASH features", logger):
        pos = pd.read_csv(DATA_FILES["pos_cash"])
        if smoke_ids:
            pos = pos[pos["SK_ID_CURR"].isin(smoke_ids)].reset_index(drop=True)