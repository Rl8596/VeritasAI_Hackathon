"""Step 1: download both datasets from the Hugging Face Hub and turn them into 256 px arrays.

    python scripts/01_prepare_data.py                 # everything (about 5 GB download)
    python scripts/01_prepare_data.py --rsna-limit 6000   # quick mode: random subset of adults
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumoshift import config as C, data  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsna-limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    if not args.no_download:
        data.download_raw()

    t = time.time()
    k = data.prepare_kermany(workers=args.workers)
    print(f"Kermany (children): {len(k)} images in {time.time() - t:.0f}s")
    print(k.groupby(["split", "label"]).size().unstack().to_string())

    t = time.time()
    r = data.prepare_rsna(workers=args.workers, limit=args.rsna_limit)
    print(f"\nRSNA (adults): {len(r)} images in {time.time() - t:.0f}s")
    print(r.groupby(["split", "cls"]).size().unstack().to_string())
    print(f"age: median {r.age[r.age_valid].median():.0f}, under 18: {(~r.is_adult).sum()}, "
          f"views: {r['view'].value_counts().to_dict()}")
    (C.PROCESSED_DIR / "DONE").write_text("ok\n")
