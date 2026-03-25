import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CREDITRISK_DATA_DIR", ROOT_DIR / "data"))

# ── Data files ───────────────────────────────────────────────────────────────
DATA_FILES = {
    "train": DATA_DIR / "application_train.csv",
    "test": DATA_DIR / "application_test.csv",
    "bureau": DATA_DIR / "bureau.csv",
    "bureau_balance": DATA_DIR / "bureau_balance.csv",
    "pos_cash": DATA_DIR / "POS_CASH_balance.csv",
    "credit_card": DATA_DIR / "credit_card_balance.csv",
    "prev_app": DATA_DIR / "previous_application.csv",
    "installments": DATA_DIR / "installments_payments.csv",
    "columns_desc": DATA_DIR / "HomeCredit_columns_description.csv",
    "sample_submission": DATA_DIR / "sample_submission.csv",
}

# ── Cross-validation ─────────────────────────────────────────────────────────
RANDOM_STATE = 42