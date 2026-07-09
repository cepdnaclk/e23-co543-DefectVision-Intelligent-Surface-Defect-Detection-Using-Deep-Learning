"""
Shared helper for writing per-category experiment results to a CSV that
survives being built up across multiple separate process runs.

evaluate.py / ablation_backbone.py / explain.py each loop over categories and
load a fresh WideResNet50 backbone per iteration. In this environment, running
all categories inside one long-lived process has intermittently segfaulted
after a couple of iterations (looks like native memory not being released
between iterations, not a bug in our own code). The practical workaround is to
invoke these scripts one category at a time (a fresh OS process per category,
so memory is fully released), which requires the CSV writer to merge new rows
into any existing file rather than overwriting it.
"""

from pathlib import Path

import pandas as pd


def save_or_merge_csv(rows: list[dict], csv_path: str | Path, key_cols: list[str]) -> pd.DataFrame:
    """
    Merge `rows` into the CSV at `csv_path`, replacing any existing row that
    matches on `key_cols` and keeping all others. Writes the combined result
    back to `csv_path`.
    """
    csv_path = Path(csv_path)
    new_df = pd.DataFrame(rows)

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    else:
        combined = new_df

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False, float_format="%.4f")
    return combined
