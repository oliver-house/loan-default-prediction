"""
Feature engineering for installments_payments.csv
"""

import numpy as np
import pandas as pd

from src.config import RECENCY_DAYS
from src.utils.helpers import reduce_mem_usage


def process_installments(ins: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installments_payments to one row per SK_ID_CURR."""
    ins = reduce_mem_usage(ins, verbose=False)

    # ── Derived columns ───────────────────────────────────────────────────────
    ins["PAYMENT_DIFF"]  = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]   # >0 = underpaid
    ins["PAYMENT_RATIO"] = ins["AMT_PAYMENT"] / (ins["AMT_INSTALMENT"] + 1)
    ins["DPD"]           = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]  # >0 = late
    ins["DPD"]           = ins["DPD"].clip(lower=0)
    ins["DBD"]           = ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]  # >0 = early
    ins["DBD"]           = ins["DBD"].clip(lower=0)
    ins["LATE_FLAG"]     = (ins["DPD"] > 0).astype(np.int8)

    # ── Numeric aggregations ──────────────────────────────────────────────────
    num_agg = {
        "NUM_INSTALMENT_VERSION":   ["nunique"],
        "DPD":                      ["max", "mean", "sum", "var"],
        "DBD":                      ["max", "mean", "sum"],
        "PAYMENT_DIFF":             ["max", "mean", "sum", "var"],
        "PAYMENT_RATIO":            ["max", "mean", "min", "var"],
        "AMT_INSTALMENT":           ["max", "mean", "sum"],
        "AMT_PAYMENT":              ["max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT":       ["max", "mean"],
        "LATE_FLAG":                ["mean", "sum"],
    }
    num_agg = {k: v for k, v in num_agg.items() if k in ins.columns}

    ins_agg = ins.groupby("SK_ID_CURR").agg(num_agg)
    ins_agg.columns = pd.Index(
        [f"INS_{col}_{stat}" for col, stat in ins_agg.columns]
    )
    ins_agg["INS_COUNT"] = ins.groupby("SK_ID_CURR").size()

    # ── On-time payment rate ──────────────────────────────────────────────────
    ins_agg["INS_ON_TIME_RATE"] = 1 - (
        ins_agg.get("INS_LATE_FLAG_mean", 0)
    )

    # ── Recent 12-month aggregations ─────────────────────────────────────────
    recent = ins[ins["DAYS_INSTALMENT"] >= RECENCY_DAYS]
    if not recent.empty:
        rec_agg = recent.groupby("SK_ID_CURR").agg({
            "DPD":           ["max", "mean"],
            "PAYMENT_DIFF":  ["max", "mean"],
            "LATE_FLAG":     ["mean", "sum"],
        })
        rec_agg.columns = pd.Index(
            [f"INS_RECENT_{col}_{stat}" for col, stat in rec_agg.columns]
        )
        ins_agg = ins_agg.join(rec_agg, how="left")

    return ins_agg.reset_index()
