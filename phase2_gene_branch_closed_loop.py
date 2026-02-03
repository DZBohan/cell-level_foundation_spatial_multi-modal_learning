"""
Phase 2: Gene Branch Closed Loop
Masked gene reconstruction with kNN local/global pooling (registration-agnostic).

This script implements "Plan A" as discussed:
- Encode each Visium spot gene vector x_i with a lightweight MLP encoder -> z_i
- Build a kNN graph over valid spots using spot coordinates
  - local neighbors: k_local
  - global neighbors: k_global
- Pool neighbor embeddings to form:
  - z_i_local: pooled from local neighbors
  - z_i_global: pooled from global neighbors
  - Delta_i = z_i_local - z_i_global
- Fuse: z_i* = MLP(concat(z_i, z_i_local, z_i_global, Delta_i))
- Self-supervised objective: masked gene reconstruction (loss on masked positions only)

Key engineering decisions aligned with your plan:
- HVG selection / normalization / log1p etc are treated as EXTERNAL preprocessing.
  Phase 2 assumes the input matrix already contains the intended gene features (e.g., HVGs).
- Graph is precomputed once in DataModule.setup() and exported as an artifact.
- "Embedding snapshot" is exported for diagnostics ONLY (collapse checks, visualization).
  It is NOT used as downstream training input (to avoid cutting gradients in Phase 5).
- Default is Mode 1 (joint training in Phase 5): downstream should load checkpoint and
  forward recompute embeddings, allowing gradients to update Phase 2 parameters.

Dependencies:
- python >= 3.10
- numpy, pandas, scipy
- torch
- pytorch-lightning or lightning
- optional: anndata (for .h5ad input)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Lightning import (works for both "pytorch_lightning" and "lightning")
try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L

from scipy.io import mmread
from scipy import sparse


# ============================================================
# 0) SOFT-CODED INPUT / OUTPUT AREA (edit this section only)
# ============================================================

@dataclass
class Phase2Paths:
    """
    Soft-coded paths for Phase 2.

    Visium input:
    - Preferred: 10x MTX bundle directory (matrix.mtx[.gz], barcodes, features, tissue_positions)
    - Optional: .h5ad cache file (must contain X and obs with coordinates)

    Outputs:
    - output_dir will contain:
        artifacts/phase2/graph/...
        artifacts/phase2/snapshots/...
        lightning_logs/...
        checkpoints/...
    """
    # ---- Identify sample ----
    sample_id: str = "HTA12_269"

    # ---- Visium gene matrix + coords (choose ONE) ----
    visium_mtx_dir: Optional[Path] = Path("/path/to/visium_mtx_bundle_dir")
    visium_h5ad: Optional[Path] = None

    # ---- Output root ----
    output_dir: Path = Path("./outputs_phase2")


@dataclass
class Phase2Config:
    """
    Soft-coded configuration for Phase 2.

    Notes:
    - This phase assumes the gene matrix already contains the desired gene features (e.g., HVGs).
    - kNN is constructed ONLY over valid spots (in_tissue==1 when available).
    - Safety checks are "hard fail" by default to prevent silent misuse.
    """
    # ---- Reproducibility ----
    seed: int = 7

    # ---- kNN graph ----
    k_local: int = 9
    k_global: int = 36

    # Hard fail if k is invalid (k >= n_valid_spots).
    hard_fail_if_k_invalid: bool = True

    # ---- Distance-weighted pooling ----
    # If False: mean pooling over neighbors.
    # If True: weights = exp(-d^2 / (2*sigma^2)) normalized per node.
    use_distance_weighted_pooling: bool = True

    # Apply weighting to which pools.
    # Options:
    # - "none": no weighted pooling (equivalent to use_distance_weighted_pooling=False)
    # - "global": weight only global pooling, local is mean
    # - "both": weight local and global
    weighted_pooling_for: str = "global"  # {"none","global","both"}

    # Sigma strategy:
    # - "fixed": use sigma_local / sigma_global as provided
    # - "median_kdist": estimate sigma from median distance to kth neighbor across valid spots
    sigma_estimator: str = "median_kdist"  # {"fixed","median_kdist"}
    sigma_local: Optional[float] = None
    sigma_global: Optional[float] = None

    # ---- Masked reconstruction ----
    mask_ratio: float = 0.3
    use_mask_indicator: bool = True
    loss_on_mask_only: bool = True  # should remain True for Plan A

    # ---- Model dims ----
    d_spot: int = 256  # spot encoder output dim (z_i)
    d_fused: int = 256  # fused embedding dim (z_i*)

    # ---- Optimization ----
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_steps: int = 200  # total optimization steps (full-graph batches)
    precision: str = "16-mixed" if torch.cuda.is_available() else "32"

    # ---- Logging / diagnostics ----
    log_every_n_steps: int = 10
    print_collapse_stats_every_n_steps: int = 50

    # ---- Export artifacts ----
    export_graph_artifacts: bool = True
    export_embedding_snapshot: bool = True
    snapshot_every_n_steps: int = 0  # 0 => only export at train end
    snapshot_use_fused_embedding: bool = True  # export z_i* if True else export z_i


# ============================================================
# 1) Utilities (seed, safe I/O, basic checks)
# ============================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def assert_no_nan_inf(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).sum().item()
        raise ValueError(f"[SafetyCheck] {name} contains NaN/Inf. bad_count={bad}")


# ============================================================
# 2) Visium loader (same spirit as Phase 0; minimal contract)
# ============================================================

@dataclass
class VisiumData:
    """
    Minimal unified representation for Phase 2.

    X: dense float32 tensor [N_spots, G] (already HVGs externally if desired)
    coords: float32 tensor [N_spots, 2] for kNN graph
    spot_ids: list[str]
    valid_mask: bool tensor [N_spots] (in_tissue==1 if available; else all True)
    """
    X: torch.Tensor
    coords: torch.Tensor
    spot_ids: list[str]
    valid_mask: torch.Tensor


def load_visium_from_mtx(visium_dir: Path, sample_id: str) -> VisiumData:
    """
    Load Visium MTX bundle (10x style).

    Expectations (typical):
    - matrix.mtx[.gz]
    - barcodes.tsv[.gz]
    - features.tsv[.gz]
    - tissue_positions.csv OR tissue_positions_list.csv

    This loader:
    - Aligns matrix rows to spot order (barcodes)
    - Extracts pxl_row / pxl_col as coordinates (x=pxl_col, y=pxl_row)
    - Uses in_tissue when available to define valid_mask

    IMPORTANT:
    - This phase assumes gene selection/normalization is external.
      If you provide raw counts here, the model will still run, but scale may be large.
    """
    visium_dir = Path(visium_dir)
    if not visium_dir.exists():
        raise FileNotFoundError(f"visium_mtx_dir not found: {visium_dir}")

    # Find files (handle .gz or non-.gz)
    def pick(cands):
        for c in cands:
            if (visium_dir / c).exists():
                return visium_dir / c
        return None

    mtx_path = pick(["matrix.mtx.gz", "matrix.mtx"])
    barcodes_path = pick(["barcodes.tsv.gz", "barcodes.tsv"])
    features_path = pick(["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"])
    tp_path = pick(["tissue_positions.csv", "tissue_positions_list.csv"])

    if mtx_path is None:
        raise FileNotFoundError(f"Cannot find matrix.mtx(.gz) in {visium_dir}")
    if barcodes_path is None:
        raise FileNotFoundError(f"Cannot find barcodes.tsv(.gz) in {visium_dir}")
    if features_path is None:
        raise FileNotFoundError(f"Cannot find features.tsv(.gz) in {visium_dir}")
    if tp_path is None:
        raise FileNotFoundError(f"Cannot find tissue_positions*.csv in {visium_dir}")

    X = mmread(str(mtx_path))
    if sparse.issparse(X):
        X = X.tocsr()
    else:
        X = sparse.csr_matrix(X)

    barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")[0].astype(str).tolist()
    features = pd.read_csv(features_path, header=None, sep="\t")

    # Determine orientation:
    # - Many 10x matrices are genes x spots. We need spots x genes.
    if X.shape[1] == len(barcodes):
        X_counts = X.transpose().tocsr()  # [spots, genes]
    elif X.shape[0] == len(barcodes):
        X_counts = X.tocsr()  # [spots, genes]
    else:
        raise ValueError(
            f"MTX shape {X.shape} does not match barcodes length {len(barcodes)}."
        )

    tp = pd.read_csv(tp_path, header=None)
    if tp.shape[1] >= 6 and tp.columns.tolist() == list(range(tp.shape[1])):
        # Standard no-header format:
        # barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
        tp.columns = ["spot_id", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"] + \
                     [f"extra_{i}" for i in range(6, tp.shape[1])]
    else:
        # If file has headers, you may need to map column names here.
        # For safety, enforce these minimal fields.
        required = {"spot_id", "pxl_row", "pxl_col"}
        if not required.issubset(set(tp.columns)):
            raise ValueError(
                f"tissue_positions file does not contain required columns {required}. "
                f"Found: {list(tp.columns)}"
            )

    tp["spot_id"] = tp["spot_id"].astype(str)

    # Align tissue_positions rows to barcodes order
    tp_map = tp.set_index("spot_id")
    missing = [b for b in barcodes if b not in tp_map.index]
    if len(missing) > 0:
        # Not fatal, but indicates mismatch between matrix and tissue_positions.
        # For safety, we hard fail (contract mismatch).
        raise ValueError(f"barcodes not found in tissue_positions: {missing[:10]} (showing up to 10)")

    tp_aligned = tp_map.loc[barcodes].reset_index()

    x = tp_aligned["pxl_col"].astype(float).values
    y = tp_aligned["pxl_row"].astype(float).values
    coords = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)

    # valid_mask
    if "in_tissue" in tp_aligned.columns:
        valid_mask = torch.tensor(tp_aligned["in_tissue"].astype(int).values == 1, dtype=torch.bool)
    else:
        valid_mask = torch.ones(len(barcodes), dtype=torch.bool)

    # Convert to dense float32 for the minimal baseline.
    # For large gene dims, this can be heavy; but Phase 2 assumes HVG-reduced matrix.
    X_dense = torch.tensor(X_counts.toarray().astype(np.float32), dtype=torch.float32)

    return VisiumData(X=X_dense, coords=coords, spot_ids=barcodes, valid_mask=valid_mask)


def load_visium_from_h5ad(h5ad_path: Path, sample_id: str) -> VisiumData:
    """
    Optional: load Visium from .h5ad.
    Contract:
    - adata.X is [N_spots, G]
    - adata.obs has coordinate columns (x/y or pxl_col/pxl_row or imagecol/imagerow)
    - adata.obs_names are spot_ids
    - adata.obs may have in_tissue (optional)
    """
    try:
        import anndata as ad
    except ImportError as e:
        raise ImportError("anndata is required to load .h5ad. Install with `pip install anndata`.") from e

    adata = ad.read_h5ad(h5ad_path)
    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr().toarray()
    else:
        X = np.asarray(X)

    X_dense = torch.tensor(X.astype(np.float32), dtype=torch.float32)

    obs = adata.obs.copy()
    spot_ids = adata.obs_names.astype(str).tolist()

    # Find coordinate columns
    xcol, ycol = None, None
    for candx, candy in [("x", "y"), ("pxl_col", "pxl_row"), ("imagecol", "imagerow")]:
        if candx in obs.columns and candy in obs.columns:
            xcol, ycol = candx, candy
            break
    if xcol is None:
        raise ValueError("Cannot find coordinate columns in adata.obs (x/y, pxl_col/pxl_row, imagecol/imagerow).")

    coords = torch.tensor(
        np.stack([obs[xcol].astype(float).values, obs[ycol].astype(float).values], axis=1),
        dtype=torch.float32
    )

    if "in_tissue" in obs.columns:
        valid_mask = torch.tensor(obs["in_tissue"].astype(int).values == 1, dtype=torch.bool)
    else:
        valid_mask = torch.ones(len(spot_ids), dtype=torch.bool)

    return VisiumData(X=X_dense, coords=coords, spot_ids=spot_ids, valid_mask=valid_mask)


# ============================================================
# 3) kNN graph builder (with safety checks + optional sigma estimation)
# ============================================================

@dataclass
class KNNGraph:
    """
    Graph artifact stored and reused downstream.

    For each node i (over ALL spots, including invalid), we store:
    - nn_idx_local[i]: indices of local neighbors (length k_local) or -1 for invalid rows
    - nn_dist_local[i]: distances to local neighbors
    - nn_idx_global[i]: indices of global neighbors (length k_global) or -1 for invalid rows
    - nn_dist_global[i]: distances to global neighbors
    - valid_mask: which spots are considered valid for graph construction
    """
    nn_idx_local: torch.Tensor   # [N, k_local] int64
    nn_dist_local: torch.Tensor  # [N, k_local] float32
    nn_idx_global: torch.Tensor  # [N, k_global] int64
    nn_dist_global: torch.Tensor # [N, k_global] float32
    valid_mask: torch.Tensor     # [N] bool


def build_knn_graph_torch(
    coords: torch.Tensor,
    valid_mask: torch.Tensor,
    k_local: int,
    k_global: int,
    hard_fail_if_k_invalid: bool = True,
) -> KNNGraph:
    """
    Build kNN neighbor lists over valid spots using torch.cdist.

    Complexity:
    - This is O(N^2) distance computation over valid spots.
    - For typical Visium N~3k-6k this is manageable for a first version.
    - For larger N, replace with FAISS / ANN / spatial indexing.

    Safety:
    - If k >= n_valid, either hard fail (recommended) or clamp.
    """
    device = coords.device
    N = coords.shape[0]
    valid_idx = torch.nonzero(valid_mask, as_tuple=True)[0]
    n_valid = valid_idx.numel()

    if hard_fail_if_k_invalid:
        if k_local >= n_valid or k_global >= n_valid:
            raise ValueError(
                f"[SafetyCheck] Invalid k: n_valid_spots={n_valid}, "
                f"k_local={k_local}, k_global={k_global}. Require k < n_valid_spots."
            )
    else:
        k_local = min(k_local, max(1, n_valid - 1))
        k_global = min(k_global, max(1, n_valid - 1))

    # Compute pairwise distances among valid spots
    vcoords = coords[valid_idx]  # [n_valid, 2]
    # torch.cdist returns [n_valid, n_valid]
    dist = torch.cdist(vcoords, vcoords, p=2)

    # Exclude self by setting diagonal to +inf
    dist.fill_diagonal_(float("inf"))

    # Get kNN indices within valid subset
    # topk with largest=False gives smallest distances
    d_local, j_local = torch.topk(dist, k=k_local, largest=False, dim=1)
    d_global, j_global = torch.topk(dist, k=k_global, largest=False, dim=1)

    # Map back to global indices
    nn_idx_local = torch.full((N, k_local), -1, dtype=torch.long, device=device)
    nn_dist_local = torch.full((N, k_local), float("inf"), dtype=torch.float32, device=device)

    nn_idx_global = torch.full((N, k_global), -1, dtype=torch.long, device=device)
    nn_dist_global = torch.full((N, k_global), float("inf"), dtype=torch.float32, device=device)

    nn_idx_local[valid_idx] = valid_idx[j_local]   # [n_valid, k_local] -> global indices
    nn_dist_local[valid_idx] = d_local.to(torch.float32)

    nn_idx_global[valid_idx] = valid_idx[j_global]
    nn_dist_global[valid_idx] = d_global.to(torch.float32)

    return KNNGraph(
        nn_idx_local=nn_idx_local,
        nn_dist_local=nn_dist_local,
        nn_idx_global=nn_idx_global,
        nn_dist_global=nn_dist_global,
        valid_mask=valid_mask.to(torch.bool),
    )


def estimate_sigma_median_kdist(nn_dist: torch.Tensor, valid_mask: torch.Tensor) -> float:
    """
    Estimate sigma from the median distance-to-kth-neighbor over valid nodes.

    nn_dist: [N, k] distances; invalid nodes may contain inf.
    We take the last column (kth neighbor distance) on valid rows, then median.
    """
    valid_rows = torch.nonzero(valid_mask, as_tuple=True)[0]
    kth = nn_dist[valid_rows, -1]  # [n_valid]
    kth = kth[torch.isfinite(kth)]
    if kth.numel() == 0:
        raise ValueError("[SafetyCheck] Cannot estimate sigma: no finite kth distances found.")
    return float(torch.median(kth).item())


# ============================================================
# 4) Pooling (mean or distance-weighted)
# ============================================================

def weighted_pool(
    z: torch.Tensor,
    nn_idx: torch.Tensor,
    nn_dist: torch.Tensor,
    sigma: float,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Distance-weighted pooling over neighbors.

    z: [N, D]
    nn_idx: [N, k] neighbor indices (global)
    nn_dist: [N, k] distances
    sigma: weight scale in the SAME coordinate unit as nn_dist
    valid_mask: [N] bool

    For invalid nodes, return zeros.

    Weight rule:
      w_ij = exp(-d_ij^2 / (2*sigma^2))
      pooled_i = sum_j w_ij * z_j / sum_j w_ij

    Note:
    - nn_idx for invalid rows may be -1; we must guard indexing.
    """
    N, D = z.shape
    k = nn_idx.shape[1]
    out = torch.zeros((N, D), dtype=z.dtype, device=z.device)

    # Only compute for valid rows
    valid_rows = torch.nonzero(valid_mask, as_tuple=True)[0]
    if valid_rows.numel() == 0:
        return out

    idx = nn_idx[valid_rows]  # [n_valid, k]
    dist = nn_dist[valid_rows]  # [n_valid, k]

    # Sanity: ensure indices are valid
    if (idx < 0).any():
        # In this implementation, valid nodes should have valid neighbor indices.
        # If this triggers, the graph contract is broken.
        bad = (idx < 0).sum().item()
        raise ValueError(f"[SafetyCheck] Found {bad} negative neighbor indices on valid rows.")

    # Gather neighbor embeddings: [n_valid, k, D]
    z_nei = z[idx]  # advanced indexing

    # Compute weights: [n_valid, k]
    sigma = max(float(sigma), 1e-8)
    w = torch.exp(-(dist ** 2) / (2.0 * sigma * sigma))
    wsum = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    w = w / wsum

    # Weighted sum
    pooled = (w.unsqueeze(-1) * z_nei).sum(dim=1)  # [n_valid, D]
    out[valid_rows] = pooled
    return out


def mean_pool(
    z: torch.Tensor,
    nn_idx: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean pooling over neighbors.

    For invalid nodes, return zeros.
    """
    N, D = z.shape
    out = torch.zeros((N, D), dtype=z.dtype, device=z.device)

    valid_rows = torch.nonzero(valid_mask, as_tuple=True)[0]
    if valid_rows.numel() == 0:
        return out

    idx = nn_idx[valid_rows]
    if (idx < 0).any():
        bad = (idx < 0).sum().item()
        raise ValueError(f"[SafetyCheck] Found {bad} negative neighbor indices on valid rows.")

    z_nei = z[idx]  # [n_valid, k, D]
    out[valid_rows] = z_nei.mean(dim=1)
    return out


# ============================================================
# 5) Dataset: full-graph batch (one sample => one batch)
# ============================================================

class FullGraphVisiumDataset(Dataset):
    """
    A dataset that returns exactly ONE item: the full spot matrix for one sample.

    Rationale:
    - Local/global pooling requires neighbor embeddings that are not guaranteed to be in a minibatch.
    - Full-graph training is the simplest and most reliable baseline for Phase 2.
    - For typical Visium spot counts (a few thousand), an MLP forward on GPU is feasible.
    - If you later need minibatching, you must implement neighbor-subgraph sampling carefully.
    """
    def __init__(self, visium: VisiumData, graph: KNNGraph):
        self.visium = visium
        self.graph = graph

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "X": self.visium.X,                 # [N, G]
            "coords": self.visium.coords,       # [N, 2]
            "spot_ids": self.visium.spot_ids,   # list[str]
            "valid_mask": self.visium.valid_mask,  # [N]
            "nn_idx_local": self.graph.nn_idx_local,
            "nn_dist_local": self.graph.nn_dist_local,
            "nn_idx_global": self.graph.nn_idx_global,
            "nn_dist_global": self.graph.nn_dist_global,
        }


def collate_full_graph(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate for full-graph dataset: just return the single dict.
    """
    assert len(batch) == 1
    return batch[0]


# ============================================================
# 6) Model: Spot encoder + fusion + masked reconstruction decoder
# ============================================================

class MLP(nn.Module):
    """
    Simple MLP utility with LayerNorm for stable training.
    """
    def __init__(self, d_in: int, d_hidden: int, d_out: int, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = d_in
        for i in range(n_layers - 1):
            layers.append(nn.Linear(d, d_hidden))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(d_hidden))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = d_hidden
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_mask(
    N: int,
    G: int,
    valid_mask: torch.Tensor,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a per-spot random gene mask: True means "masked position".

    Only valid spots participate in masking; invalid spots are fully unmasked
    (they are excluded from the loss anyway).
    """
    mask = torch.zeros((N, G), dtype=torch.bool, device=device)
    valid_rows = torch.nonzero(valid_mask, as_tuple=True)[0]
    if valid_rows.numel() == 0:
        return mask

    # For each valid row, sample approx mask_ratio*G genes
    # This is simple and reliable; for very large G, you may optimize later.
    n_mask = int(round(mask_ratio * G))
    n_mask = max(1, min(G, n_mask))

    for i in valid_rows.tolist():
        idx = torch.randperm(G, device=device)[:n_mask]
        mask[i, idx] = True
    return mask


class Phase2GeneLightningModule(L.LightningModule):
    """
    Phase 2 Lightning module implementing Plan A.

    Forward:
    - x_masked (+ optional mask indicator) -> encoder -> z
    - pool z to z_local and z_global using kNN graph
    - delta = z_local - z_global
    - fuse -> z_star
    - decode -> x_hat
    - loss = MSE over masked gene positions only (valid spots only)

    Diagnostics:
    - collapse checks (variance, norm)
    - NaN/Inf checks
    - optional snapshot export
    """
    def __init__(self, cfg: Phase2Config, gene_dim: int, output_dir: Path):
        super().__init__()
        self.cfg = cfg
        self.gene_dim = gene_dim
        self.output_dir = Path(output_dir)

        # Encoder input dim:
        # - base gene vector: G
        # - optional mask indicator: G (1 indicates masked)
        enc_in = gene_dim + (gene_dim if cfg.use_mask_indicator else 0)

        self.spot_encoder = MLP(
            d_in=enc_in,
            d_hidden=max(256, cfg.d_spot),
            d_out=cfg.d_spot,
            n_layers=3,
            dropout=0.0
        )

        # Fusion: concat(z, z_local, z_global, delta) => 4 * d_spot
        self.fusion = MLP(
            d_in=4 * cfg.d_spot,
            d_hidden=max(256, cfg.d_fused),
            d_out=cfg.d_fused,
            n_layers=2,
            dropout=0.0
        )

        # Decoder: fused embedding -> reconstruct gene vector
        self.decoder = MLP(
            d_in=cfg.d_fused,
            d_hidden=max(256, cfg.d_fused),
            d_out=gene_dim,
            n_layers=2,
            dropout=0.0
        )

        # We store sigmas after estimation (in setup logic via on_fit_start)
        self._sigma_local: Optional[float] = None
        self._sigma_global: Optional[float] = None

        self._step_count = 0
        self._last_snapshot_path: Optional[Path] = None

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def on_fit_start(self) -> None:
        """
        Estimate sigma if requested, using the graph distances seen in the first batch.
        We do this here so sigma is logged and saved with the run.
        """
        # The actual distances are in the batch; we will estimate sigma in the first training_step.
        pass

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        X = batch["X"]  # [N, G]
        valid_mask = batch["valid_mask"]  # [N]
        nn_idx_local = batch["nn_idx_local"]
        nn_dist_local = batch["nn_dist_local"]
        nn_idx_global = batch["nn_idx_global"]
        nn_dist_global = batch["nn_dist_global"]

        device = X.device
        N, G = X.shape

        # Safety checks on input
        assert_no_nan_inf(X, "X")

        # Estimate sigma (one-time) if needed
        if self.cfg.use_distance_weighted_pooling and self.cfg.weighted_pooling_for != "none":
            if self.cfg.sigma_estimator == "median_kdist" and (self._sigma_local is None or self._sigma_global is None):
                # Estimate from median kth-neighbor distances over valid nodes
                if self._sigma_local is None:
                    self._sigma_local = estimate_sigma_median_kdist(nn_dist_local, valid_mask)
                if self._sigma_global is None:
                    self._sigma_global = estimate_sigma_median_kdist(nn_dist_global, valid_mask)

                # Log estimated sigmas
                self.log("sigma_local", float(self._sigma_local), prog_bar=True)
                self.log("sigma_global", float(self._sigma_global), prog_bar=True)

            elif self.cfg.sigma_estimator == "fixed":
                # Use fixed sigmas (must be provided)
                if self._sigma_local is None:
                    if self.cfg.sigma_local is None:
                        raise ValueError("sigma_estimator='fixed' requires cfg.sigma_local.")
                    self._sigma_local = float(self.cfg.sigma_local)
                if self._sigma_global is None:
                    if self.cfg.sigma_global is None:
                        raise ValueError("sigma_estimator='fixed' requires cfg.sigma_global.")
                    self._sigma_global = float(self.cfg.sigma_global)

        # ---------------------------
        # Masked reconstruction setup
        # ---------------------------
        gene_mask = make_mask(
            N=N, G=G, valid_mask=valid_mask, mask_ratio=self.cfg.mask_ratio, device=device
        )  # True => masked positions

        # Build x_masked
        X_masked = X.clone()
        X_masked[gene_mask] = 0.0

        # Optional mask indicator (explicitly tells the model what was masked)
        if self.cfg.use_mask_indicator:
            mask_indicator = gene_mask.to(X.dtype)  # 1.0 on masked positions
            enc_in = torch.cat([X_masked, mask_indicator], dim=1)  # [N, 2G]
        else:
            enc_in = X_masked  # [N, G]

        # ---------------------------
        # Spot encoding: z_i
        # ---------------------------
        z = self.spot_encoder(enc_in)  # [N, d_spot]
        assert_no_nan_inf(z, "z")

        # ---------------------------
        # Pooling: z_local, z_global
        # ---------------------------
        # Decide whether to use distance weighting
        use_weight = self.cfg.use_distance_weighted_pooling and self.cfg.weighted_pooling_for != "none"

        # Local pooling
        if use_weight and self.cfg.weighted_pooling_for in ("both",):
            if self._sigma_local is None:
                raise ValueError("sigma_local not initialized.")
            z_local = weighted_pool(z, nn_idx_local, nn_dist_local, self._sigma_local, valid_mask)
        else:
            z_local = mean_pool(z, nn_idx_local, valid_mask)

        # Global pooling
        if use_weight and self.cfg.weighted_pooling_for in ("global", "both"):
            if self._sigma_global is None:
                raise ValueError("sigma_global not initialized.")
            z_global = weighted_pool(z, nn_idx_global, nn_dist_global, self._sigma_global, valid_mask)
        else:
            z_global = mean_pool(z, nn_idx_global, valid_mask)

        delta = z_local - z_global  # [N, d_spot]

        # ---------------------------
        # Fusion: z_i*
        # ---------------------------
        fused_in = torch.cat([z, z_local, z_global, delta], dim=1)  # [N, 4*d_spot]
        z_star = self.fusion(fused_in)  # [N, d_fused]
        assert_no_nan_inf(z_star, "z_star")

        # ---------------------------
        # Decode: reconstruct genes
        # ---------------------------
        X_hat = self.decoder(z_star)  # [N, G]
        assert_no_nan_inf(X_hat, "X_hat")

        # ---------------------------
        # Loss: masked positions only (valid spots only)
        # ---------------------------
        valid_rows = torch.nonzero(valid_mask, as_tuple=True)[0]
        if valid_rows.numel() == 0:
            # No valid spots: training signal is undefined. Hard fail.
            raise ValueError("[SafetyCheck] No valid spots found (valid_mask is all False).")

        if self.cfg.loss_on_mask_only:
            # Loss is computed only on masked entries for valid spots.
            m = gene_mask[valid_rows]  # [n_valid, G]
            target = X[valid_rows][m]
            pred = X_hat[valid_rows][m]

            # If mask_ratio is extremely small, m could be empty by chance; guard it.
            if target.numel() == 0:
                raise ValueError("[SafetyCheck] Empty masked target. Increase mask_ratio or check valid_mask.")
            loss = nn.functional.mse_loss(pred, target)
        else:
            # Not recommended for Plan A; included for completeness.
            loss = nn.functional.mse_loss(X_hat[valid_rows], X[valid_rows])

        # ---------------------------
        # Diagnostics / logging
        # ---------------------------
        self.log("loss_masked_recon", loss.detach(), prog_bar=True)

        if self.cfg.print_collapse_stats_every_n_steps > 0 and (self._step_count % self.cfg.print_collapse_stats_every_n_steps == 0):
            self._log_collapse_stats(z_star if self.cfg.snapshot_use_fused_embedding else z)

        # Optional snapshot export mid-training
        if self.cfg.export_embedding_snapshot and self.cfg.snapshot_every_n_steps > 0:
            if self._step_count > 0 and (self._step_count % self.cfg.snapshot_every_n_steps == 0):
                self._export_snapshot(batch, z, z_star, step=self._step_count)

        self._step_count += 1
        return loss

    def _log_collapse_stats(self, z_any: torch.Tensor) -> None:
        """
        Simple collapse / stability diagnostics:
        - mean norm
        - feature variance
        """
        with torch.no_grad():
            zn = torch.norm(z_any, dim=1)  # [N]
            feat_var = torch.var(z_any, dim=0).mean()  # scalar
            self.log("emb_norm_mean", zn.mean(), prog_bar=False)
            self.log("emb_norm_std", zn.std(), prog_bar=False)
            self.log("emb_feat_var_mean", feat_var, prog_bar=False)

    def on_train_end(self) -> None:
        """
        Export embedding snapshot at the end of training (diagnostic only).

        This export does NOT freeze anything by itself.
        It is simply a forward output of the final checkpoint state.
        Downstream training (Phase 5) should load the checkpoint and recompute embeddings
        during forward passes to allow gradients to update Phase 2 parameters.
        """
        if not self.cfg.export_embedding_snapshot:
            return

        # We cannot access the batch here directly; we rely on a stored reference if any.
        # For simplicity and robustness, we re-run snapshot export using the graph/data
        # stored by the datamodule via trainer, if available.
        dm = getattr(self.trainer, "datamodule", None)
        if dm is None or getattr(dm, "cached_batch", None) is None:
            # If this triggers, you can still export snapshots manually by running a forward pass.
            print("[Phase2] Warning: datamodule cached_batch not found; skipping end-of-train snapshot export.")
            return

        batch = dm.cached_batch
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        X = batch["X"]
        valid_mask = batch["valid_mask"]
        nn_idx_local = batch["nn_idx_local"]
        nn_dist_local = batch["nn_dist_local"]
        nn_idx_global = batch["nn_idx_global"]
        nn_dist_global = batch["nn_dist_global"]

        N, G = X.shape

        # Use unmasked input for snapshot (diagnostic embedding on "clean" input)
        if self.cfg.use_mask_indicator:
            zeros_mask = torch.zeros((N, G), dtype=X.dtype, device=X.device)
            enc_in = torch.cat([X, zeros_mask], dim=1)
        else:
            enc_in = X

        with torch.no_grad():
            z = self.spot_encoder(enc_in)
            # Pooling (same logic as training)
            use_weight = self.cfg.use_distance_weighted_pooling and self.cfg.weighted_pooling_for != "none"

            if use_weight and self.cfg.weighted_pooling_for in ("both",):
                z_local = weighted_pool(z, nn_idx_local, nn_dist_local, float(self._sigma_local), valid_mask)
            else:
                z_local = mean_pool(z, nn_idx_local, valid_mask)

            if use_weight and self.cfg.weighted_pooling_for in ("global", "both"):
                z_global = weighted_pool(z, nn_idx_global, nn_dist_global, float(self._sigma_global), valid_mask)
            else:
                z_global = mean_pool(z, nn_idx_global, valid_mask)

            delta = z_local - z_global
            z_star = self.fusion(torch.cat([z, z_local, z_global, delta], dim=1))

        self._export_snapshot(batch, z, z_star, step=self._step_count, is_final=True)

    def _export_snapshot(self, batch: Dict[str, Any], z: torch.Tensor, z_star: torch.Tensor, step: int, is_final: bool = False) -> None:
        """
        Export embedding snapshot for diagnostics only.

        Stored content:
        - embeddings (z or z_star)
        - spot_ids
        - valid_mask
        - config metadata
        """
        out_root = self.output_dir / "artifacts" / "phase2" / "snapshots"
        ensure_dir(out_root)

        emb = z_star if self.cfg.snapshot_use_fused_embedding else z
        emb = emb.detach().cpu()

        payload = {
            "sample_id": getattr(self.trainer.datamodule, "sample_id", "unknown"),
            "step": int(step),
            "is_final": bool(is_final),
            "embedding_kind": "z_star" if self.cfg.snapshot_use_fused_embedding else "z",
            "embeddings": emb,  # torch tensor [N, D]
            "spot_ids": batch["spot_ids"],
            "valid_mask": batch["valid_mask"].detach().cpu(),
            "cfg": asdict(self.cfg),
        }

        name = f"embedding_snapshot_{'final' if is_final else f'step{step}'}_{payload['embedding_kind']}.pt"
        path = out_root / name
        torch.save(payload, path)
        self._last_snapshot_path = path
        print(f"[Phase2] Saved embedding snapshot: {path}")


# ============================================================
# 7) DataModule: load visium, build graph, export artifacts
# ============================================================

class Phase2DataModule(L.LightningDataModule):
    """
    Wires:
    - load Visium gene matrix + coords
    - build kNN graph (local/global)
    - export graph artifacts (optional)
    - provide full-graph DataLoader

    Also caches the full batch so the model can export final snapshots safely.
    """
    def __init__(self, paths: Phase2Paths, cfg: Phase2Config):
        super().__init__()
        self.paths = paths
        self.cfg = cfg

        self.sample_id = paths.sample_id
        self.visium: Optional[VisiumData] = None
        self.graph: Optional[KNNGraph] = None
        self.dataset: Optional[FullGraphVisiumDataset] = None

        # Cached batch for end-of-train snapshot export
        self.cached_batch: Optional[Dict[str, Any]] = None

    def setup(self, stage: Optional[str] = None):
        set_seed(self.cfg.seed)

        # Load Visium
        if self.paths.visium_h5ad is not None:
            self.visium = load_visium_from_h5ad(self.paths.visium_h5ad, sample_id=self.sample_id)
        elif self.paths.visium_mtx_dir is not None:
            self.visium = load_visium_from_mtx(self.paths.visium_mtx_dir, sample_id=self.sample_id)
        else:
            raise ValueError("You must set either visium_mtx_dir or visium_h5ad.")

        # Build graph on CPU first for safety, then move tensors to device in training
        coords = self.visium.coords.cpu()
        valid_mask = self.visium.valid_mask.cpu()

        self.graph = build_knn_graph_torch(
            coords=coords,
            valid_mask=valid_mask,
            k_local=self.cfg.k_local,
            k_global=self.cfg.k_global,
            hard_fail_if_k_invalid=self.cfg.hard_fail_if_k_invalid,
        )

        # Export graph artifacts
        if self.cfg.export_graph_artifacts:
            self._export_graph(self.graph, self.visium, self.paths.output_dir)

        self.dataset = FullGraphVisiumDataset(self.visium, self.graph)

        # Cache the batch for snapshot export
        self.cached_batch = self.dataset[0]

    def train_dataloader(self):
        assert self.dataset is not None
        return DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_full_graph,
            pin_memory=True,
        )

    def _export_graph(self, graph: KNNGraph, visium: VisiumData, outdir: Path) -> None:
        """
        Export graph artifact for downstream reuse (Phase 4/5).

        This graph is a deterministic contract output:
        - neighbor indices + distances
        - valid mask
        - spot_ids mapping
        """
        out_root = Path(outdir) / "artifacts" / "phase2" / "graph"
        ensure_dir(out_root)

        payload = {
            "sample_id": self.sample_id,
            "k_local": int(self.cfg.k_local),
            "k_global": int(self.cfg.k_global),
            "coords_kind": "pxl_col/pxl_row (or h5ad coords)",  # update if you change loader
            "spot_ids": visium.spot_ids,
            "valid_mask": graph.valid_mask.cpu(),
            "nn_idx_local": graph.nn_idx_local.cpu(),
            "nn_dist_local": graph.nn_dist_local.cpu(),
            "nn_idx_global": graph.nn_idx_global.cpu(),
            "nn_dist_global": graph.nn_dist_global.cpu(),
            "cfg": asdict(self.cfg),
        }

        path = out_root / "spot_knn_graph.pt"
        torch.save(payload, path)
        print(f"[Phase2] Saved graph artifact: {path}")

        # Also save a small JSON summary for quick inspection
        summary = {
            "sample_id": self.sample_id,
            "N_spots": int(len(visium.spot_ids)),
            "n_valid_spots": int(graph.valid_mask.sum().item()),
            "k_local": int(self.cfg.k_local),
            "k_global": int(self.cfg.k_global),
        }
        with open(out_root / "graph_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


# ============================================================
# 8) Main entry: run Phase 2
# ============================================================

def main():
    # --------------------------
    # Edit these values in the soft-coded section above.
    # --------------------------
    paths = Phase2Paths(
        sample_id="HTA12_269",
        visium_mtx_dir=Path("/path/to/visium_mtx_bundle_dir"),
        visium_h5ad=None,
        output_dir=Path("./outputs_phase2"),
    )

    cfg = Phase2Config(
        k_local=9,
        k_global=36,
        use_distance_weighted_pooling=True,
        weighted_pooling_for="global",
        sigma_estimator="median_kdist",
        mask_ratio=0.3,
        max_steps=200,
    )

    ensure_dir(paths.output_dir)

    dm = Phase2DataModule(paths=paths, cfg=cfg)
    dm.setup()

    assert dm.visium is not None
    gene_dim = int(dm.visium.X.shape[1])

    model = Phase2GeneLightningModule(cfg=cfg, gene_dim=gene_dim, output_dir=paths.output_dir)

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=cfg.max_steps,
        log_every_n_steps=cfg.log_every_n_steps,
        precision=cfg.precision,
        enable_checkpointing=True,
        default_root_dir=str(paths.output_dir),
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()


# ============================================================
# 9) DOWNSTREAM USAGE NOTES (Phase 2 -> Phase 3/4/5/6/7)
# ============================================================

"""
Downstream usage notes (how Phase 2 artifacts are consumed later):

A) What Phase 2 exports
1) Checkpoint (trainable):
   - Location: <output_dir>/checkpoints/...
   - Content: spot_encoder + fusion + decoder (+ optimizer state if saved by Lightning)
   - Downstream (Phase 4/5): load checkpoint, forward recompute embeddings, allow gradients.
     This is required for "Mode 1" joint training where InfoNCE updates Phase 2 parameters.

2) Graph artifact (deterministic contract):
   - Location: <output_dir>/artifacts/phase2/graph/spot_knn_graph.pt
   - Contains: spot_ids, valid_mask, nn_idx_local/global, nn_dist_local/global
   - Downstream: reuse the exact same graph for pooling in Phase 4/5.
     This avoids graph drift and improves reproducibility.

3) Embedding snapshot (diagnostic only; NOT a training input):
   - Location: <output_dir>/artifacts/phase2/snapshots/embedding_snapshot_*.pt
   - Purpose:
       * collapse checks (variance, norm distribution)
       * visualization (UMAP/tSNE)
       * quick sanity-check that representations are structured
   - IMPORTANT:
       * Using snapshot embeddings as Phase 5 training inputs would cut gradients and
         effectively freeze the Phase 2 encoder. Do NOT do this in your mainline.
       * Mainline Phase 5 should always load the Phase 2 checkpoint and forward recompute embeddings.

B) Level semantics (naming to avoid confusion)
- P1 = cell-scale (H&E)  [Phase 1, image branch]
- P2 = RA-scale (H&E patch scale aligned to protein RA aggregation radius)  [Phase 5 InfoNCE L2]
- P3 = spot-scale (H&E patch scale aligned to Visium spot diameter)         [Phase 5 InfoNCE L3]
- Phase 2 uses "spot neighborhoods" in the GENE space:
    * local kNN (k_local)
    * global kNN (k_global)
  These are NOT the same as P2 "RA-scale neighborhood" and should be named explicitly
  as "spot-local" and "spot-global" neighborhoods.

C) How Phase 5 should use Phase 2 (Mode 1: joint training)
- Load Phase 2 checkpoint -> forward gene encoder to produce z_i / z_i* each step
- Compute InfoNCE with matched H&E embeddings at the appropriate level (likely L3)
- Backprop through the gene encoder (Phase 2) so alignment can refine gene representations

D) Recommended ablation switches for later phases
- freeze_gene_encoder: if True, stop gradients to Phase 2 encoder (not mainline)
- use_precomputed_gene_embeddings: if True, feed snapshot embeddings as inputs (cuts gradients)
  This is mainly for debugging or resource constraints, not your mainline plan.
"""

