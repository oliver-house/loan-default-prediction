"""Generate selected_features.json from existing feature_importances.csv"""

import json
import pandas as pd
from src.config import PREDICTIONS_DIR

IMPORTANCE_THRESHOLD = 0.99

imp_df   = pd.read_csv(PREDICTIONS_DIR / "feature_importances.csv")
cumsum   = imp_df["ensemble"].cumsum()
n        = int((cumsum < IMPORTANCE_THRESHOLD).sum()) + 1
selected   = imp_df["feature"].iloc[:n].tolist()
unselected = imp_df["feature"].iloc[n:].tolist()

print(f"Selected {len(selected)}/{len(imp_df)} features ({IMPORTANCE_THRESHOLD:.0%} cumulative importance)")

out_path = PREDICTIONS_DIR / "selected_features.json"
with open(out_path, "w") as f:
    json.dump(selected, f, indent=2)
print(f"Saved to {out_path}")

drop_path = PREDICTIONS_DIR / "unselected_features.json"
with open(drop_path, "w") as f:
    json.dump(unselected, f, indent=2)
print(f"Saved to {drop_path}")
