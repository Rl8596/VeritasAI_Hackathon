"""Download, prepare and load the two chest X-ray datasets.

Kermany  = children aged 1-5, one hospital in Guangzhou, China  (the "tutorial" dataset)
RSNA     = mostly adults, NIH Clinical Center, USA              (our reality check)

Both are stored the same way: one uint8 array [N, 256, 256] plus a CSV with one
row per image, in the same order as the array.
"""
import io
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import config as C
from .preprocess import normalize, to_square_gray

# --------------------------------------------------------------------------
# Download (only needed on a fresh machine, e.g. Colab)
# --------------------------------------------------------------------------

def download_raw(cache_dir: Path = C.CACHE_DIR) -> None:
    """Fetch the raw files from the Hugging Face Hub if they are not there yet."""
    from huggingface_hub import hf_hub_download, snapshot_download

    cache_dir = Path(cache_dir)
    if not list((cache_dir / "kermany").rglob("*.parquet")):
        snapshot_download(C.KERMANY_REPO, repo_type="dataset", allow_patterns="data/*.parquet",
                          local_dir=cache_dir / "kermany")
    if not list((cache_dir / "rsna").glob("*.zip")):
        hf_hub_download(C.RSNA_REPO, C.RSNA_ZIP, repo_type="dataset", local_dir=cache_dir / "rsna")
    mapping = cache_dir / "rsna" / "rsna_nih_mapping.json"
    if not mapping.exists():                      # RSNA's own table: challenge id -> original NIH file name
        import urllib.request
        urllib.request.urlretrieve(C.RSNA_NIH_MAPPING_URL, mapping)


# --------------------------------------------------------------------------
# Kermany (children)
# --------------------------------------------------------------------------

def _kermany_patient(filename: str, official_split: str) -> tuple[str, str]:
    """Return (patient_id, subtype) parsed from the original file name.

    Pneumonia files look like  person100_bacteria_475.jpeg
    Normal files look like     IM-0011-0001-0002.jpeg  or  NORMAL2-IM-0173-0001.jpeg
    The official test folder uses its own numbering, so ids are namespaced by it.
    """
    space = "test" if official_split == "test" else "trainval"
    m = re.match(r"person(\d+)_(bacteria|virus)_", filename)
    if m:
        return f"{space}_P{int(m.group(1))}", m.group(2)
    m = re.match(r"(?:NORMAL2-)?IM-(\d+)-", filename)
    if m:
        return f"{space}_N{int(m.group(1))}", "normal"
    return f"{space}_U{filename}", "unknown"


def _decode_kermany(args):
    raw, = args
    img = Image.open(io.BytesIO(raw))
    w, h = img.size
    return to_square_gray(img), w, h


def prepare_kermany(cache_dir: Path = C.CACHE_DIR, workers: int = 4) -> pd.DataFrame:
    import pyarrow.parquet as pq

    out_dir = Path(cache_dir) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((Path(cache_dir) / "kermany").rglob("*.parquet"))
    assert files, "Kermany parquet files not found, run download_raw() first"

    rows, arrays = [], []
    with ProcessPoolExecutor(workers) as pool:
        for f in files:
            official = f.name.split("-")[0].replace("validation", "val")
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=64):
                recs = batch.to_pylist()
                decoded = list(pool.map(_decode_kermany, [(r["image"]["bytes"],) for r in recs]))
                for r, (arr, w, h) in zip(recs, decoded):
                    name = r["image"]["path"]
                    pid, subtype = _kermany_patient(name, official)
                    rows.append(dict(filename=name, label=int(r["label"]), subtype=subtype, patient_id=pid,
                                     official_split=official, orig_w=w, orig_h=h))
                    arrays.append(arr)

    meta = pd.DataFrame(rows)
    # Our own split: the official validation set has only 16 images, so we merge it
    # with the official training set and hold out whole PATIENTS for validation.
    # The official test set stays untouched.
    from sklearn.model_selection import StratifiedGroupKFold

    meta["split"] = "test"
    pool_idx = np.flatnonzero(meta.official_split != "test")
    sgkf = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=C.SEED)
    tr, va = next(sgkf.split(pool_idx, meta.label.values[pool_idx], meta.patient_id.values[pool_idx]))
    meta.loc[pool_idx[tr], "split"] = "train"
    meta.loc[pool_idx[va], "split"] = "val"
    meta.insert(0, "idx", np.arange(len(meta)))

    np.save(out_dir / "kermany_256.npy", np.stack(arrays))
    meta.to_csv(out_dir / "kermany_meta.csv", index=False)
    return meta


# --------------------------------------------------------------------------
# RSNA (mostly adults)
# --------------------------------------------------------------------------

def _decode_rsna_chunk(args):
    zip_path, names = args
    import pydicom

    out = []
    with zipfile.ZipFile(zip_path) as z:
        for n in names:
            ds = pydicom.dcmread(io.BytesIO(z.read(n)))
            arr = ds.pixel_array
            if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                arr = arr.max() - arr
            if arr.dtype != np.uint8:
                arr = (arr.astype(np.float32) / max(float(arr.max()), 1.0) * 255).astype(np.uint8)
            img = to_square_gray(Image.fromarray(arr))
            age = str(getattr(ds, "PatientAge", "")).strip()
            out.append((Path(n).stem, img,
                        int(age) if age.isdigit() else -1,
                        str(getattr(ds, "PatientSex", "")).strip(),
                        str(getattr(ds, "ViewPosition", "")).strip()))
    return out


def prepare_rsna(cache_dir: Path = C.CACHE_DIR, workers: int = 4, limit: int | None = None) -> pd.DataFrame:
    """`limit` keeps only a random subset of that many images (quick mode for Colab)."""
    out_dir = Path(cache_dir) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted((Path(cache_dir) / "rsna").glob("*.zip"))
    assert zips, "RSNA zip not found, run download_raw() first"
    zip_path = zips[0]

    with zipfile.ZipFile(zip_path) as z:
        labels = pd.read_csv(z.open("stage_2_train_labels.csv"))
        classes = pd.read_csv(z.open("stage_2_detailed_class_info.csv")).drop_duplicates("patientId")
        names = sorted(n for n in z.namelist() if n.startswith("stage_2_train_images/") and n.endswith(".dcm"))
    if limit:
        rng = np.random.default_rng(C.SEED)
        names = sorted(rng.choice(names, size=min(limit, len(names)), replace=False).tolist())

    chunks = [names[i:i + 200] for i in range(0, len(names), 200)]
    recs = []
    with ProcessPoolExecutor(workers) as pool:
        for part in pool.map(_decode_rsna_chunk, [(str(zip_path), c) for c in chunks]):
            recs.extend(part)

    meta = pd.DataFrame([(r[0], r[2], r[3], r[4]) for r in recs], columns=["patient_id", "age", "sex", "view"])
    per_patient = labels.groupby("patientId").agg(label=("Target", "max"), n_boxes=("Target", "sum")).reset_index()
    meta = meta.merge(per_patient, left_on="patient_id", right_on="patientId", how="left").drop(columns="patientId")
    meta = meta.merge(classes.rename(columns={"patientId": "patient_id", "class": "cls"}), on="patient_id", how="left")
    meta["age_valid"] = meta.age.between(0, 100)          # a handful of ages are typos (e.g. 148)
    meta["is_adult"] = meta.age_valid & (meta.age >= C.ADULT_AGE)

    meta = _split_rsna(meta, Path(cache_dir))
    meta.insert(0, "idx", np.arange(len(meta)))

    np.save(out_dir / "rsna_256.npy", np.stack([r[1] for r in recs]))
    meta.to_csv(out_dir / "rsna_meta.csv", index=False)
    boxes = labels[labels.Target == 1].rename(columns={"patientId": "patient_id"})
    boxes = boxes[boxes.patient_id.isin(meta.patient_id)]
    boxes.to_csv(out_dir / "rsna_boxes.csv", index=False)   # radiologist boxes, in 1024 px coordinates
    return meta


def _split_rsna(meta: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """Patient-level split: adapt (60%) = adult images a model may learn from,
    val (10%) = tuning, test (30%) = final numbers only.

    The challenge's "patientId" is really an exam id: the same person can appear many
    times. RSNA publishes a table that maps each exam back to the NIH file name, which
    starts with the real patient number. We split by that number, so no person is on
    both sides of a split.
    """
    import json

    from sklearn.model_selection import StratifiedGroupKFold

    mapping = Path(cache_dir) / "rsna" / "rsna_nih_mapping.json"
    if mapping.exists():
        m = pd.DataFrame(json.loads(mapping.read_text()))
        nih = dict(zip(m.subset_img_id, m.img_id.str.split("_").str[0]))
        meta["nih_patient"] = meta.patient_id.map(nih).fillna(meta.patient_id)
    else:
        print("WARNING: NIH mapping not found, falling back to an exam-level split (patients may overlap)")
        meta["nih_patient"] = meta.patient_id
    strat = (meta.cls + "_" + meta["view"]).values
    folds = np.zeros(len(meta), dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=C.SEED)
    for f, (_, te) in enumerate(sgkf.split(strat, strat, meta.nih_patient.values)):
        folds[te] = f
    meta["split"] = np.where(folds < 3, "test", np.where(folds == 3, "val", "adapt"))
    return meta


def resplit_rsna(cache_dir: Path = C.CACHE_DIR) -> pd.DataFrame:
    """Redo only the split of an already prepared RSNA set (image order stays the same)."""
    p = Path(cache_dir) / "processed" / "rsna_meta.csv"
    meta = _split_rsna(pd.read_csv(p).drop(columns=["split", "nih_patient"], errors="ignore"), Path(cache_dir))
    meta.to_csv(p, index=False)
    return meta


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_kermany(cache_dir: Path = C.CACHE_DIR):
    """-> (X uint8 [5856, 256, 256] memory-mapped, meta DataFrame). Children aged 1-5."""
    p = Path(cache_dir) / "processed"
    return np.load(p / "kermany_256.npy", mmap_mode="r"), pd.read_csv(p / "kermany_meta.csv")


def load_rsna(cache_dir: Path = C.CACHE_DIR):
    """-> (X uint8 [N, 256, 256] memory-mapped, meta DataFrame). Mostly adults."""
    p = Path(cache_dir) / "processed"
    return np.load(p / "rsna_256.npy", mmap_mode="r"), pd.read_csv(p / "rsna_meta.csv")


def load_rsna_boxes(cache_dir: Path = C.CACHE_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(cache_dir) / "processed" / "rsna_boxes.csv")


class XrayDataset(torch.utils.data.Dataset):
    """Serves (image tensor [3, 224, 224], label, source index) from a stored uint8 array.

    train=True adds light augmentation (small shifts, zooms, rotations, brightness
    and contrast changes). No left-right flips: the heart is on one side.
    """

    def __init__(self, X, indices, labels, train: bool = False):
        self.X, self.indices, self.labels, self.train = X, np.asarray(indices), np.asarray(labels), train
        if train:
            from torchvision.transforms import v2
            self.aug = v2.Compose([
                v2.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.92, 1.08)),
                v2.RandomResizedCrop(C.IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
            ])

    def __len__(self):
        return len(self.indices)

    # On Windows every DataLoader worker is a new process that receives a pickled copy of this object.
    # A memory-mapped array pickles as its whole contents (about 2 GB per worker for the adult images),
    # so workers get the file path instead and open the file themselves.
    def __getstate__(self):
        state = self.__dict__.copy()
        path = getattr(self.X, "filename", None)
        if isinstance(self.X, np.memmap) and path is not None and np.load(path, mmap_mode="r").shape == self.X.shape:
            state["X"], state["_X_path"] = None, str(path)
        return state

    def __setstate__(self, state):
        path = state.pop("_X_path", None)
        self.__dict__.update(state)
        if path is not None:
            self.X = np.load(path, mmap_mode="r")

    def __getitem__(self, i):
        t = torch.from_numpy(np.array(self.X[self.indices[i]]))[None]      # uint8 [1, 256, 256]
        if self.train:
            t = self.aug(t)
        else:
            o = (t.shape[-1] - C.IMG_SIZE) // 2
            t = t[:, o:o + C.IMG_SIZE, o:o + C.IMG_SIZE]
        return normalize(t[None])[0], float(self.labels[i]), int(self.indices[i])
