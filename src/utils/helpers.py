import logging
import time
from contextlib import contextmanager

import pandas as pd

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

@contextmanager
def timer(label: str, logger: logging.Logger | None = None):
    """Context manager that logs elapsed time for a block of code."""
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    msg = f"{label} done in {elapsed:.1f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)

def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True):
    """
    One-hot encode all object columns.
    Returns (encoded_df, list_of_new_column_names).
    """
    original_columns = list(df.columns)
    categorical_columns = df.select_dtypes(["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns