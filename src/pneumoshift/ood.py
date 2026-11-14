"""Out-of-distribution warning: "this X-ray is unlike anything I was trained on".

Class-conditional Mahalanobis distance on the model's embedding (Lee et al., NeurIPS 2018):
fit one mean per class and one shared covariance on the training embeddings; the
score of a new image is its distance to the nearest class mean. Large = unfamiliar.
"""
from pathlib import Path

import numpy as np


class MahalanobisOOD:
    def __init__(self):
        self.means = None        # [n_classes, D]
        self.precision = None    # [D, D]
        self.thresholds = {}     # e.g. {"p95": 31.2, "p99": 38.0} from in-distribution validation images
        self.ref_scores = None   # sorted validation scores, to turn a score into a percentile

    def fit(self, emb: np.ndarray, labels: np.ndarray) -> "MahalanobisOOD":
        from sklearn.covariance import LedoitWolf

        emb = np.asarray(emb, dtype=np.float64)
        labels = np.asarray(labels).astype(int)
        classes = np.unique(labels)
        self.means = np.stack([emb[labels == c].mean(0) for c in classes])
        centred = emb - self.means[np.searchsorted(classes, labels)]
        self.precision = LedoitWolf().fit(centred).precision_
        return self

    def score(self, emb: np.ndarray) -> np.ndarray:
        emb = np.atleast_2d(np.asarray(emb, dtype=np.float64))
        d = emb[:, None, :] - self.means[None]                       # [N, n_classes, D]
        m = np.einsum("ncd,de,nce->nc", d, self.precision, d)
        return np.sqrt(np.maximum(m.min(axis=1), 0))

    def calibrate(self, emb_val: np.ndarray) -> "MahalanobisOOD":
        """Set warning thresholds from in-distribution validation images the model was NOT trained on."""
        s = np.sort(self.score(emb_val))
        self.ref_scores = s
        self.thresholds = {"p95": float(np.percentile(s, 95)), "p99": float(np.percentile(s, 99))}
        return self

    def percentile(self, score) -> np.ndarray:
        """Share of validation images that looked MORE familiar than this one (0..100)."""
        return 100.0 * np.searchsorted(self.ref_scores, np.atleast_1d(score)) / len(self.ref_scores)

    def save(self, path) -> None:
        np.savez_compressed(path, means=self.means.astype(np.float32), precision=self.precision.astype(np.float32),
                            ref_scores=np.asarray(self.ref_scores, dtype=np.float32),
                            p95=self.thresholds["p95"], p99=self.thresholds["p99"])

    @classmethod
    def load(cls, path) -> "MahalanobisOOD":
        z = np.load(Path(path))
        o = cls()
        o.means, o.precision = z["means"].astype(np.float64), z["precision"].astype(np.float64)
        o.ref_scores = z["ref_scores"]
        o.thresholds = {"p95": float(z["p95"]), "p99": float(z["p99"])}
        return o
