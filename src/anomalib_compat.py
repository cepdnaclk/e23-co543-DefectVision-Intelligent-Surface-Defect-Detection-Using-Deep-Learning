"""
Compatibility shim for a pandas/anomalib version interaction bug.

anomalib's MVTecAD datamodule filters its samples DataFrame with
``samples.split == split`` where ``split`` is a ``Split`` enum member (e.g.
``Split.TRAIN``). ``Split`` is defined as ``class Split(str, Enum)``, so a
plain Python comparison like ``Split.TRAIN == "train"`` is True — but with the
pandas version pinned by this project's anomalib version, vectorized Series
comparison against the enum member (``pd.Series([...]) == Split.TRAIN``)
returns all False instead of matching "train" rows. The net effect: the
MVTecAD datamodule silently returns a 0-length train/test split for every
category, which crashes anything that calls ``engine.fit()`` (empty
DataLoader) and *silently* produces zero predictions for anything that calls
``engine.predict()``/``engine.test()`` (no crash, just empty results).

This patches anomalib's dataset-building function to coerce the split
argument to a plain string before filtering, side-stepping the pandas/enum
comparison entirely. Call `patch_split_enum_bug()` once, before constructing
any `anomalib.data.MVTecAD` datamodule.
"""

import anomalib.data.datasets.image.mvtecad as _mvtecad_module

_original_make_mvtec_ad_dataset = _mvtecad_module.make_mvtec_ad_dataset
_patched = False


def patch_split_enum_bug() -> None:
    """Idempotently patch anomalib's MVTecAD sample filtering (see module docstring)."""
    global _patched
    if _patched:
        return

    def _make_mvtec_ad_dataset_patched(root, split=None, extensions=None):
        if hasattr(split, "value"):
            split = split.value
        return _original_make_mvtec_ad_dataset(root, split=split, extensions=extensions)

    _mvtecad_module.make_mvtec_ad_dataset = _make_mvtec_ad_dataset_patched
    _patched = True
