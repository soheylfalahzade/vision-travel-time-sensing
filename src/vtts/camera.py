"""Camera observation model with emulated re-identification embeddings."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vtts.simulate import TrafficData


@dataclass
class CameraConfig:
    """Observation noise (documented assumptions, not measurements)."""

    p_miss: float = 0.08           # probability that a passing vehicle is not detected
    t_sigma_s: float = 0.3         # timestamp noise
    dim: int = 32
    n_classes: int = 6             # appearance clusters (make/model/colour groups)
    class_weight: float = 0.6      # shared component of same-class vehicles (higher = harder)
    cam_noise: tuple[float, float] = (0.4, 1.0)   # per-camera viewpoint noise range


@dataclass
class Observations:
    """Camera detections (one row per detected passage)."""

    cam: np.ndarray
    t: np.ndarray
    t_true: np.ndarray
    vid: np.ndarray
    in_e: np.ndarray
    out_e: np.ndarray
    emb: np.ndarray


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_observations(data: TrafficData, cams: list[int], cfg: CameraConfig,
                      rng: np.random.Generator) -> Observations:
    """Detect passages at camera nodes and attach noisy appearance embeddings."""
    n_veh = len(data.vtype)
    centres = _unit(rng.normal(size=(cfg.n_classes, cfg.dim)))
    cls = rng.integers(cfg.n_classes, size=n_veh)
    eta = _unit(rng.normal(size=(n_veh, cfg.dim)))
    w = cfg.class_weight
    latent = _unit(w * centres[cls] + np.sqrt(1 - w * w) * eta)
    cam_sigma = {c: rng.uniform(*cfg.cam_noise) for c in cams}
    p = data.passages
    keep = np.isin(p[:, 1].astype(int), cams) & (rng.random(len(p)) >= cfg.p_miss)
    p = p[keep]
    vid = p[:, 0].astype(int)
    sig = np.array([cam_sigma[int(c)] for c in p[:, 1]])[:, None]
    emb = _unit(latent[vid] + sig * rng.normal(size=(len(p), cfg.dim)) / np.sqrt(cfg.dim))
    return Observations(p[:, 1].astype(int), p[:, 2] + rng.normal(0, cfg.t_sigma_s, len(p)),
                        p[:, 2], vid, p[:, 3].astype(int), p[:, 4].astype(int), emb)
