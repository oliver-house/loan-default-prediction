import argparse

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
        pass

if __name__ == "__main__":
    main()