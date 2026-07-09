"""
Download MVTec AD dataset for the 3 required categories.

Method 1: anomalib's MVTecAD datamodule (auto-download)
Method 2: huggingface_hub snapshot download
Method 3: Manual download instructions

Target structure:
  data/mvtec_ad/
    bottle/train/good/, test/<defect>/, ground_truth/<defect>/
    hazelnut/...
    carpet/...
"""

import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CATEGORIES = ["bottle", "hazelnut", "carpet"]
DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_ad"

# MVTec AD direct download URLs (individual category tarballs)
MVTEC_URLS = {
    "bottle": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951845/hazelnut.tar.xz",
    "carpet": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937078-1629951672/carpet.tar.xz",
}


def check_category_exists(category):
    """Check if a category is already downloaded."""
    cat_dir = DATA_ROOT / category
    required = [cat_dir / "train" / "good", cat_dir / "test"]
    return all(d.exists() and any(d.iterdir()) for d in required)


def download_with_url(category):
    """Download category tarball from MVTec's mydrive mirror."""
    import tarfile
    import urllib.request

    url = MVTEC_URLS.get(category)
    if not url:
        print(f"  [FAIL] No URL for {category}")
        return False

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    tar_path = DATA_ROOT / f"{category}.tar.xz"

    try:
        print(f"  Downloading {category}.tar.xz ...")
        urllib.request.urlretrieve(url, tar_path)
        print(f"  Extracting...")
        with tarfile.open(tar_path, "r:xz") as tar:
            tar.extractall(path=str(DATA_ROOT))
        tar_path.unlink()

        if check_category_exists(category):
            print(f"  [OK] {category} downloaded and extracted")
            return True
        else:
            print(f"  [FAIL] Extraction succeeded but expected files not found")
            return False
    except Exception as e:
        print(f"  [FAIL] Direct download failed: {e}")
        if tar_path.exists():
            tar_path.unlink()
        return False


def download_with_anomalib(category):
    """Download using anomalib's MVTecAD datamodule."""
    try:
        from anomalib.data import MVTecAD

        from src.anomalib_compat import patch_split_enum_bug
        patch_split_enum_bug()

        print(f"  Trying anomalib auto-download...")
        datamodule = MVTecAD(root=str(DATA_ROOT), category=category, num_workers=0)
        datamodule.prepare_data()
        datamodule.setup()

        if check_category_exists(category):
            print(f"  [OK] {category} downloaded via anomalib")
            return True
        else:
            print(f"  [FAIL] anomalib download did not produce expected files")
            return False
    except Exception as e:
        print(f"  [FAIL] anomalib download failed: {e}")
        return False


def print_manual_instructions():
    """Print manual download fallback."""
    print(f"""
{'='*70}
MANUAL DOWNLOAD INSTRUCTIONS
{'='*70}

If automatic download fails:

1. Visit: https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Accept the license (CC BY-NC-SA 4.0)
3. Download: bottle.tar.xz, hazelnut.tar.xz, carpet.tar.xz
4. Extract each into: {DATA_ROOT}

   tar -xf bottle.tar.xz -C {DATA_ROOT}/
   tar -xf hazelnut.tar.xz -C {DATA_ROOT}/
   tar -xf carpet.tar.xz -C {DATA_ROOT}/

Verify structure:
  {DATA_ROOT}/bottle/train/good/  (should have ~200 PNG images)
  {DATA_ROOT}/bottle/test/        (multiple subdirs)
  {DATA_ROOT}/bottle/ground_truth/
""")


def main():
    print("=" * 60)
    print("MVTec AD Dataset Download")
    print(f"Categories: {', '.join(CATEGORIES)}")
    print(f"Target: {DATA_ROOT}")
    print("=" * 60)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    success, failed = [], []

    for category in CATEGORIES:
        print(f"\n[{category}]")

        if check_category_exists(category):
            print(f"  [OK] Already exists, skipping")
            success.append(category)
            continue

        # Try direct URL first (most reliable)
        if download_with_url(category):
            success.append(category)
            continue

        # Fallback to anomalib
        if download_with_anomalib(category):
            success.append(category)
            continue

        failed.append(category)

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    if success:
        print(f"  OK:     {', '.join(success)}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        print_manual_instructions()
        sys.exit(1)
    else:
        print("  All categories ready!")


if __name__ == "__main__":
    main()
