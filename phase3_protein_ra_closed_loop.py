# phase3_protein_ra_representation_closed_loop.py
"""
Phase 3: Local Protein Context (RA) Representation Learning (closed loop)
========================================================================

What this phase does (as finalized in our discussion)
-----------------------------------------------------
Phase 3 constructs a *coordinate-anchored* protein representation for each H&E-derived cell anchor by:
1) Reusing the **exact same H&E cell anchors** as Phase 1 / Phase 0 contract:
      he_cells_table columns must include: ["sample_id", "he_cell_id", "x", "y"]
   - NOTE: naming must be identical across scripts. We do NOT rename these columns here.

2) Consuming a **precomputed CODEX / protein cell table** (Phase 3 does NOT do CODEX segmentation):
      codex_cells_table must include:
        - ["sample_id", "codex_cell_id", "x", "y"] + protein feature columns
   - This table is treated as an external preprocessing artifact.
   - Phase 3 never touches DAPI masks / segmentation logic.

3) Performing **Radius-based Aggregation (RA)** around each H&E anchor coordinate (x, y):
      RA_i = Agg( { protein_cell_features_k | dist((x_k,y_k),(x_i,y_i)) <= r } )

4) Producing a "RA representation space" indexed by (sample_id, he_cell_id):
      output: ra_emb_i ∈ R^d

Design philosophy (contract-first)
----------------------------------
- The identity of an anchor is strictly (sample_id, he_cell_id).
- The anchor coordinates are strictly "x, y" from he_cells_table (same as Phase 1).
- CODEX / protein cells use a different id name: codex_cell_id (never he_cell_id).
- Phase 3 output is RA representations; it performs NO cross-modal alignment.
- Downstream Phase 4/5/6 will map embeddings to shared / contrastive spaces.

Why this "closed loop" script exists
------------------------------------
- It validates that the Phase 3 pipeline runs end-to-end:
  input tables -> RA computation -> optional self-supervised learning -> embeddings saved.
- It is intentionally lightweight and modular:
  - RA computation is deterministic, contract-driven, and debuggable.
  - Self-supervised objective is *optional* and can be disabled.
- It preserves a strict separation:
  - RA (data-level aggregation) is not conflated with alignment losses (later phases).

Dependencies
------------
- python >= 3.10
- numpy, pandas
- torch
- (optional) scipy: for cKDTree radius queries (recommended for speed)

Outputs
-------
- out_dir/ra_embeddings/ra_emb.parquet:
    columns: [sample_id, he_cell_id] + ra_emb_{0..d-1}
- out_dir/checkpoints/protein_encoder.pt (if training is enabled)
- out_dir/qc/ra_stats.json

Downstream usage
----------------
- Phase 4 will load:
    - H&E embeddings keyed by (sample_id, he_cell_id) from Phase 1/4
    - RA embeddings keyed by (sample_id, he_cell_id) from Phase 3
  and then apply per-modality adapters to enter shared latent space.

"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Optional fast radius queries
try:
    from scipy.spatial import cKDTree  # type: ignore
    _HAS_CKDTREE = True
except Exception:
    _HAS_CKDTREE = False


# ============================================================
# 0) SOFT-CODED INPUT AREA (edit only here)
# ============================================================

@dataclass
class Phase3InputPaths:
    """
    Soft-coded input paths for Phase 3.

    Contract-critical inputs:
    - he_cells_table: MUST match Phase 1 / Phase 0 schema and names.
      required cols: ["sample_id","he_cell_id","x","y"]

    - codex_cells_table: precomputed protein cell table (segmentation is out-of-scope for Phase 3).
      required cols (minimum): ["sample_id","codex_cell_id","x","y"] + protein feature cols

    Optional:
    - protein_feature_cols: if None, we auto-detect feature columns as:
        all columns excluding ["sample_id","codex_cell_id","x","y"].
      For maximum reproducibility, you can explicitly list them.

    """
    sample_id: str = "HTA12_269"

    # Phase 1 / Phase 0 anchor table (MUST keep same column names)
    he_cells_table: Path = Path("/path/to/he_cells.parquet")

    # External preprocessing artifact: CODEX/protein cell table
    codex_cells_table: Path = Path("/path/to/codex_cells.parquet")

    # If None, auto-detect from codex_cells_table
    protein_feature_cols: Optional[List[str]] = None


@dataclass
class Phase3Config:
    """
    Soft-coded parameters for Phase 3.

    Core RA parameters:
    - ra_radius: radius in the same coordinate system as x,y in both tables.
      If x,y are pixel coordinates, radius is pixels.
      If x,y are microns, radius is microns.
      (You must keep units consistent across modalities.)

    Representation:
    - input_feat_dim: inferred from protein_feature_cols
    - ra_emb_dim: output embedding dimension

    Training (optional):
    - enable_training: if False, we only compute deterministic RA mean features and save them.
      If True, we train a small MLP encoder with a masked-feature reconstruction objective.

    """
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- RA ----
    ra_radius: float = 30.0  # units must match your x,y coordinate units
    min_cells_in_radius: int = 1  # if fewer, we fallback to zeros (or nearest neighbor if enabled)

    # ---- Fallback behavior when no cells within radius ----
    # If True, use the nearest protein cell feature when radius query is empty.
    # This is a practical safeguard for sparse regions / edge artifacts.
    fallback_to_nearest_if_empty: bool = True

    # ---- Embedding ----
    ra_emb_dim: int = 256
    encoder_hidden_dim: int = 512
    encoder_num_layers: int = 2
    use_layernorm: bool = True

    # ---- Training (self-supervised; optional) ----
    enable_training: bool = True
    batch_size: int = 256
    num_workers: int = 4
    max_steps: int = 2000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_amp: bool = True
    grad_accum_steps: int = 1

    # Masked-feature reconstruction objective
    mask_ratio: float = 0.3           # fraction of protein features to mask
    recon_loss_weight: float = 1.0    # MSE on masked dims
    emb_var_reg_weight: float = 0.0   # optional variance regularizer (0 disables)

    # ---- Logging / saving ----
    print_every_n_steps: int = 100
    save_every_n_steps: int = 500

    out_dir: Path = Path("./phase3_outputs")
    save_ra_mean_features: bool = True  # save the raw RA-mean features (pre-encoder) for debugging


# ============================================================
# 1) Utilities: seeding / I/O / validation
# ============================================================

REQUIRED_HE_CELLS_COLS = ["sample_id", "he_cell_id", "x", "y"]
REQUIRED_CODEX_BASE_COLS = ["sample_id", "codex_cell_id", "x", "y"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Table not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in [".csv", ".tsv"]:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def validate_cols(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing required columns: {missing}\nFound: {list(df.columns)}")


def infer_protein_feature_cols(codex_df: pd.DataFrame) -> List[str]:
    """
    Infer protein feature columns by excluding base columns.
    You should pin this list in config for full reproducibility.
    """
    base = set(REQUIRED_CODEX_BASE_COLS)
    feats = [c for c in codex_df.columns if c not in base]
    if len(feats) == 0:
        raise ValueError("No protein feature columns detected in codex_cells_table.")
    return feats


def ensure_float32_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    """
    Convert selected columns to a float32 numpy matrix.
    """
    mat = df[cols].to_numpy()
    if not np.issubdtype(mat.dtype, np.number):
        raise ValueError("Protein feature columns must be numeric.")
    return mat.astype(np.float32, copy=False)


# ============================================================
# 2) Radius-based aggregation (RA)
# ============================================================

class RadiusAggregator:
    """
    RA engine:
    - Builds a spatial index over CODEX/protein cell coordinates.
    - Queries all protein cells within ra_radius for each H&E anchor coordinate.
    - Aggregates their protein feature vectors into a single RA feature vector.

    Speed:
    - If scipy is available: use cKDTree (fast).
    - Otherwise: fallback to brute-force distance checks (OK for small demos, slow for large N).
    """

    def __init__(
        self,
        codex_xy: np.ndarray,          # shape [N,2]
        codex_feats: np.ndarray,       # shape [N,F]
        ra_radius: float,
        min_cells_in_radius: int,
        fallback_to_nearest_if_empty: bool,
    ) -> None:
        assert codex_xy.ndim == 2 and codex_xy.shape[1] == 2
        assert codex_feats.ndim == 2 and codex_feats.shape[0] == codex_xy.shape[0]

        self.codex_xy = codex_xy.astype(np.float32, copy=False)
        self.codex_feats = codex_feats.astype(np.float32, copy=False)
        self.ra_radius = float(ra_radius)
        self.min_cells = int(min_cells_in_radius)
        self.fallback_to_nearest = bool(fallback_to_nearest_if_empty)

        self._tree = None
        if _HAS_CKDTREE:
            self._tree = cKDTree(self.codex_xy)

    def query_indices_within_radius(self, q_xy: np.ndarray) -> List[np.ndarray]:
        """
        Return indices of protein cells within radius for each query point.
        """
        if self._tree is not None:
            # scipy cKDTree returns python lists of indices
            inds_list = self._tree.query_ball_point(q_xy, r=self.ra_radius)
            return [np.asarray(inds, dtype=np.int64) for inds in inds_list]

        # Brute-force fallback
        out: List[np.ndarray] = []
        for q in q_xy:
            d2 = np.sum((self.codex_xy - q[None, :]) ** 2, axis=1)
            inds = np.where(d2 <= (self.ra_radius ** 2))[0].astype(np.int64)
            out.append(inds)
        return out

    def nearest_index(self, q_xy: np.ndarray) -> np.ndarray:
        """
        Return nearest protein cell index for each query point.
        """
        if self._tree is not None:
            d, idx = self._tree.query(q_xy, k=1)
            return np.asarray(idx, dtype=np.int64)

        # Brute-force fallback
        idxs = []
        for q in q_xy:
            d2 = np.sum((self.codex_xy - q[None, :]) ** 2, axis=1)
            idxs.append(int(np.argmin(d2)))
        return np.asarray(idxs, dtype=np.int64)

    def aggregate(self, q_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate protein features for each query coordinate.

        Returns:
        - ra_mean_feats: [M,F] float32
        - ra_counts: [M] int32 number of cells aggregated
        """
        inds_list = self.query_indices_within_radius(q_xy)
        F = self.codex_feats.shape[1]

        ra_mean = np.zeros((q_xy.shape[0], F), dtype=np.float32)
        counts = np.zeros((q_xy.shape[0],), dtype=np.int32)

        if self.fallback_to_nearest:
            nearest = self.nearest_index(q_xy)

        for i, inds in enumerate(inds_list):
            if inds.size >= self.min_cells:
                feats = self.codex_feats[inds]  # [K,F]
                ra_mean[i] = feats.mean(axis=0)
                counts[i] = inds.size
            else:
                # Empty / too few cells
                if self.fallback_to_nearest:
                    ra_mean[i] = self.codex_feats[nearest[i]]
                    counts[i] = 1
                else:
                    ra_mean[i] = 0.0
                    counts[i] = 0

        return ra_mean, counts


# ============================================================
# 3) Optional encoder + masked reconstruction (self-supervised)
# ============================================================

class MLPEncoder(nn.Module):
    """
    Small MLP encoder: RA mean features -> RA embedding.

    This is intentionally lightweight:
    - It is *not* cross-modal.
    - It does *not* require labels.
    - It is used only to learn a stable protein representation space within Phase 3.

    You can disable training and just save RA mean features if desired.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int, use_layernorm: bool):
        super().__init__()
        assert num_layers >= 1

        layers: List[nn.Module] = []
        d_in = in_dim
        for li in range(num_layers - 1):
            layers.append(nn.Linear(d_in, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedReconHead(nn.Module):
    """
    Reconstruction head: RA embedding -> reconstruct original RA mean feature vector.

    This supports a simple masked-feature reconstruction objective:
    - Randomly mask a subset of input feature dimensions.
    - Encode masked input -> embedding -> reconstruct full feature vector.
    - Compute MSE on masked dimensions only.

    This is a stable, lightweight, self-supervised objective for tabular protein features.
    """

    def __init__(self, emb_dim: int, feat_dim: int, hidden_dim: int, use_layernorm: bool):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim),
        ]
        if use_layernorm:
            # Optional LN on output can stabilize early steps when feat scales differ.
            layers.insert(2, nn.LayerNorm(hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def make_feature_mask(batch_size: int, feat_dim: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    """
    Create a boolean mask of shape [B,F] where True indicates a masked dimension.
    """
    # Bernoulli mask
    m = torch.rand((batch_size, feat_dim), device=device) < float(mask_ratio)
    # Ensure at least one masked dim per sample (practical stability)
    # If a row has no masked dims, force-mask one random dim.
    row_sum = m.sum(dim=1)
    if (row_sum == 0).any():
        idx = torch.where(row_sum == 0)[0]
        rand_cols = torch.randint(low=0, high=feat_dim, size=(idx.numel(),), device=device)
        m[idx, rand_cols] = True
    return m


def embedding_variance_regularizer(z: torch.Tensor) -> torch.Tensor:
    """
    Optional variance regularizer: encourages non-collapsed embeddings.
    Simple version: mean variance across dims (maximize it).
    We return a loss term to *minimize*, so use negative variance.
    """
    var = z.var(dim=0, unbiased=False).mean()
    return -var


# ============================================================
# 4) Dataset: compute RA mean features once, then train on them
# ============================================================

class Phase3RADataset(Dataset):
    """
    Phase 3 dataset:
    - Indexed by H&E anchors (sample_id, he_cell_id, x, y).
    - Each item returns:
        {
          sample_id, he_cell_id,
          ra_mean_feats, ra_count
        }

    Important:
    - This dataset does NOT do any CODEX segmentation.
    - It consumes a precomputed CODEX cell table.
    """

    def __init__(
        self,
        he_cells: pd.DataFrame,
        ra_mean_feats: np.ndarray,   # [M,F]
        ra_counts: np.ndarray,       # [M]
        protein_feature_cols: List[str],
    ) -> None:
        super().__init__()
        validate_cols(he_cells, REQUIRED_HE_CELLS_COLS, "he_cells")
        assert ra_mean_feats.shape[0] == len(he_cells)
        assert ra_counts.shape[0] == len(he_cells)

        self.he_cells = he_cells.reset_index(drop=True)
        self.ra_mean = ra_mean_feats.astype(np.float32, copy=False)
        self.ra_counts = ra_counts.astype(np.int32, copy=False)
        self.protein_feature_cols = protein_feature_cols

    def __len__(self) -> int:
        return len(self.he_cells)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.he_cells.iloc[idx]
        return {
            "sample_id": str(r["sample_id"]),
            "he_cell_id": str(r["he_cell_id"]),
            "ra_mean_feats": torch.from_numpy(self.ra_mean[idx]).float(),  # [F]
            "ra_count": int(self.ra_counts[idx]),
        }


def collate_phase3(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["sample_id"] = [b["sample_id"] for b in batch]
    out["he_cell_id"] = [b["he_cell_id"] for b in batch]
    out["ra_count"] = torch.tensor([b["ra_count"] for b in batch], dtype=torch.int32)
    out["ra_mean_feats"] = torch.stack([b["ra_mean_feats"] for b in batch], dim=0)  # [B,F]
    return out


# ============================================================
# 5) Training loop (plain PyTorch; no Lightning required)
# ============================================================

@torch.no_grad()
def compute_basic_stats(x: torch.Tensor) -> Dict[str, float]:
    """
    Basic statistics for debugging / collapse detection:
    - mean norm
    - mean variance across dimensions
    """
    norm = torch.norm(x, dim=-1).mean().item()
    var = x.var(dim=0, unbiased=False).mean().item()
    return {"norm_mean": float(norm), "var_mean": float(var)}


def train_phase3_encoder(
    dl: DataLoader,
    encoder: nn.Module,
    recon_head: nn.Module,
    cfg: Phase3Config,
) -> Tuple[nn.Module, nn.Module]:
    """
    Train a protein encoder with masked reconstruction (self-supervised).

    Notes:
    - This is intentionally minimal and stable.
    - It does not assume any cross-modality pairing.
    - It learns a representation space for RA-aggregated protein context vectors.
    """
    device = torch.device(cfg.device)
    encoder = encoder.to(device)
    recon_head = recon_head.to(device)

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(recon_head.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    encoder.train()
    recon_head.train()

    step = 0
    while step < cfg.max_steps:
        for batch in dl:
            if step >= cfg.max_steps:
                break

            x = batch["ra_mean_feats"].to(device)  # [B,F]
            B, F = x.shape

            # ---- build masked input ----
            mask = make_feature_mask(B, F, cfg.mask_ratio, device=device)  # [B,F] bool
            x_masked = x.clone()
            x_masked[mask] = 0.0

            # ---- forward ----
            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                z = encoder(x_masked)              # [B,D]
                x_hat = recon_head(z)              # [B,F]

                # MSE on masked dims only (stable objective)
                mse = (x_hat - x) ** 2
                loss_recon = mse[mask].mean()

                loss = cfg.recon_loss_weight * loss_recon

                # Optional anti-collapse regularizer on embeddings
                if cfg.emb_var_reg_weight > 0:
                    loss_var = embedding_variance_regularizer(z)
                    loss = loss + cfg.emb_var_reg_weight * loss_var

            # ---- backward ----
            scaler.scale(loss / cfg.grad_accum_steps).backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            # ---- logging ----
            if cfg.print_every_n_steps > 0 and (step % cfg.print_every_n_steps == 0):
                with torch.no_grad():
                    st_z = compute_basic_stats(z.detach())
                print(
                    f"[Phase3][train] step={step:06d} "
                    f"loss={loss.item():.6f} recon={loss_recon.item():.6f} "
                    f"z_norm={st_z['norm_mean']:.4f} z_var={st_z['var_mean']:.6f}"
                )

            # ---- checkpointing ----
            if cfg.save_every_n_steps > 0 and (step > 0) and (step % cfg.save_every_n_steps == 0):
                ckpt_dir = Path(cfg.out_dir) / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "step": step,
                        "encoder_state_dict": encoder.state_dict(),
                        "recon_head_state_dict": recon_head.state_dict(),
                        "cfg": cfg.__dict__,
                    },
                    ckpt_dir / f"protein_encoder_step{step:06d}.pt",
                )

            step += 1

    return encoder, recon_head


@torch.no_grad()
def encode_all_ra_embeddings(
    dl: DataLoader,
    encoder: nn.Module,
    cfg: Phase3Config,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Encode all RA mean feature vectors into RA embeddings.

    Output table contract:
    - sample_id, he_cell_id
    - ra_emb_000 ... ra_emb_{D-1}

    This table is the canonical Phase 3 artifact used downstream (Phase 4/5/6).
    """
    device = torch.device(cfg.device)
    encoder = encoder.to(device)
    encoder.eval()

    rows: List[Dict[str, Any]] = []
    all_z: List[torch.Tensor] = []

    for batch in dl:
        x = batch["ra_mean_feats"].to(device)  # [B,F]
        z = encoder(x)                         # [B,D]
        z_cpu = z.detach().cpu()
        all_z.append(z_cpu)

        for i in range(z_cpu.shape[0]):
            r: Dict[str, Any] = {
                "sample_id": batch["sample_id"][i],
                "he_cell_id": batch["he_cell_id"][i],
            }
            # Store embedding dimensions as explicit columns for easy parquet usage.
            for j in range(z_cpu.shape[1]):
                r[f"ra_emb_{j:03d}"] = float(z_cpu[i, j].item())
            rows.append(r)

    Z = torch.cat(all_z, dim=0)
    stats = compute_basic_stats(Z)
    df = pd.DataFrame(rows)
    return df, stats


# ============================================================
# 6) Main: wire everything together
# ============================================================

def main():
    # ----------------
    # Instantiate soft-coded inputs
    # ----------------
    paths = Phase3InputPaths(
        sample_id="HTA12_269",
        he_cells_table=Path("/path/to/he_cells.parquet"),
        codex_cells_table=Path("/path/to/codex_cells.parquet"),
        protein_feature_cols=None,  # set explicitly for maximum reproducibility
    )

    cfg = Phase3Config(
        ra_radius=30.0,
        ra_emb_dim=256,
        enable_training=True,
        max_steps=2000,
        batch_size=256,
        out_dir=Path("./phase3_outputs"),
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    # ----------------
    # Load and validate tables (contract-critical)
    # ----------------
    he_cells = read_table(paths.he_cells_table)
    validate_cols(he_cells, REQUIRED_HE_CELLS_COLS, "he_cells_table")

    codex_cells = read_table(paths.codex_cells_table)
    validate_cols(codex_cells, REQUIRED_CODEX_BASE_COLS, "codex_cells_table")

    # Filter to a single sample for this run (consistent with Phase 1 demo style)
    sid = str(paths.sample_id)
    he_cells = he_cells[he_cells["sample_id"].astype(str) == sid].copy()
    codex_cells = codex_cells[codex_cells["sample_id"].astype(str) == sid].copy()

    if len(he_cells) == 0:
        raise ValueError(f"No H&E anchors found for sample_id={sid}. Check he_cells_table.")
    if len(codex_cells) == 0:
        raise ValueError(f"No CODEX/protein cells found for sample_id={sid}. Check codex_cells_table.")

    # Determine protein feature columns
    if paths.protein_feature_cols is None:
        protein_feature_cols = infer_protein_feature_cols(codex_cells)
    else:
        protein_feature_cols = list(paths.protein_feature_cols)
        # Verify user-specified columns exist
        missing = [c for c in protein_feature_cols if c not in codex_cells.columns]
        if missing:
            raise ValueError(f"protein_feature_cols not found in codex_cells_table: {missing}")

    # Build coordinate arrays
    he_xy = he_cells[["x", "y"]].to_numpy(dtype=np.float32)              # [M,2] anchors
    codex_xy = codex_cells[["x", "y"]].to_numpy(dtype=np.float32)        # [N,2] protein cells
    codex_feats = ensure_float32_matrix(codex_cells, protein_feature_cols)  # [N,F]

    # ----------------
    # Compute deterministic RA mean features (the core Phase 3 operation)
    # ----------------
    ra_engine = RadiusAggregator(
        codex_xy=codex_xy,
        codex_feats=codex_feats,
        ra_radius=cfg.ra_radius,
        min_cells_in_radius=cfg.min_cells_in_radius,
        fallback_to_nearest_if_empty=cfg.fallback_to_nearest_if_empty,
    )

    ra_mean_feats, ra_counts = ra_engine.aggregate(he_xy)

    # Save raw RA mean features for debugging (optional but recommended early on)
    if cfg.save_ra_mean_features:
        out_raw = Path(cfg.out_dir) / "ra_mean_features"
        out_raw.mkdir(parents=True, exist_ok=True)
        raw_df = he_cells[["sample_id", "he_cell_id"]].copy()
        raw_df["ra_count"] = ra_counts
        for j, c in enumerate(protein_feature_cols):
            raw_df[f"ra_mean_{c}"] = ra_mean_feats[:, j]
        raw_df.to_parquet(out_raw / "ra_mean_features.parquet", index=False)

    # ----------------
    # Build dataset/dataloader on RA mean features
    # ----------------
    ds = Phase3RADataset(
        he_cells=he_cells,
        ra_mean_feats=ra_mean_feats,
        ra_counts=ra_counts,
        protein_feature_cols=protein_feature_cols,
    )

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_phase3,
        drop_last=True if cfg.enable_training else False,
    )

    # ----------------
    # Optional training: learn RA embedding space
    # ----------------
    feat_dim = len(protein_feature_cols)
    encoder = MLPEncoder(
        in_dim=feat_dim,
        out_dim=cfg.ra_emb_dim,
        hidden_dim=cfg.encoder_hidden_dim,
        num_layers=cfg.encoder_num_layers,
        use_layernorm=cfg.use_layernorm,
    )
    recon_head = MaskedReconHead(
        emb_dim=cfg.ra_emb_dim,
        feat_dim=feat_dim,
        hidden_dim=cfg.encoder_hidden_dim,
        use_layernorm=cfg.use_layernorm,
    )

    if cfg.enable_training:
        encoder, recon_head = train_phase3_encoder(dl, encoder, recon_head, cfg)

        # Save final checkpoint
        ckpt_dir = Path(cfg.out_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": cfg.max_steps,
                "encoder_state_dict": encoder.state_dict(),
                "recon_head_state_dict": recon_head.state_dict(),
                "protein_feature_cols": protein_feature_cols,
                "cfg": cfg.__dict__,
            },
            ckpt_dir / "protein_encoder_final.pt",
        )
        print(f"[Phase3] Saved final checkpoint to: {ckpt_dir / 'protein_encoder_final.pt'}")
    else:
        print("[Phase3] enable_training=False: skipping encoder training (deterministic RA only).")

    # ----------------
    # Encode all anchors into RA embeddings (canonical artifact)
    # ----------------
    dl_eval = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_phase3,
        drop_last=False,
    )

    ra_emb_df, ra_stats = encode_all_ra_embeddings(dl_eval, encoder, cfg)

    out_emb = Path(cfg.out_dir) / "ra_embeddings"
    out_emb.mkdir(parents=True, exist_ok=True)
    ra_emb_path = out_emb / "ra_emb.parquet"
    ra_emb_df.to_parquet(ra_emb_path, index=False)

    # ----------------
    # Save QC stats
    # ----------------
    qc_dir = Path(cfg.out_dir) / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    qc = {
        "sample_id": sid,
        "num_he_anchors": int(len(he_cells)),
        "num_codex_cells": int(len(codex_cells)),
        "ra_radius": float(cfg.ra_radius),
        "min_cells_in_radius": int(cfg.min_cells_in_radius),
        "fallback_to_nearest_if_empty": bool(cfg.fallback_to_nearest_if_empty),
        "protein_feature_dim": int(feat_dim),
        "ra_emb_dim": int(cfg.ra_emb_dim),
        "has_ckdtree": bool(_HAS_CKDTREE),
        "ra_emb_stats": ra_stats,
        "ra_count_mean": float(np.mean(ra_counts)),
        "ra_count_min": int(np.min(ra_counts)),
        "ra_count_max": int(np.max(ra_counts)),
        "ra_count_zero_frac": float(np.mean(ra_counts == 0)),
    }
    with open(qc_dir / "ra_stats.json", "w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2)

    print(f"[Phase3] Saved RA embeddings to: {ra_emb_path}")
    print(f"[Phase3] Saved QC stats to: {qc_dir / 'ra_stats.json'}")
    print("[Phase3] Done.")


if __name__ == "__main__":
    main()


# ============================================================
# 7) DOWNSTREAM USAGE NOTES (Phase 3 -> downstream phases)
# ============================================================

"""
Phase 3 outputs (artifacts) and downstream usage
------------------------------------------------

Canonical artifact:
(1) ra_embeddings/ra_emb.parquet
    - Schema:
        sample_id: str
        he_cell_id: str
        ra_emb_000 ... ra_emb_{D-1}: float
    - Contract:
        - (sample_id, he_cell_id) must match Phase 1 anchors exactly.
        - This is the ONLY join key expected by Phase 4/5/6.

Optional debugging artifacts:
(2) ra_mean_features/ra_mean_features.parquet
    - Useful for verifying RA behavior and sanity-checking aggregation.
    - Contains:
        sample_id, he_cell_id, ra_count, ra_mean_{protein_feature_name...}

(3) qc/ra_stats.json
    - Contains basic run metadata and embedding distribution stats.

How Phase 3 feeds into the 8-phase plan
----------------------------------------
- Phase 3 does NOT perform CODEX segmentation.
  It consumes a precomputed codex_cells_table (cell centroid + protein feature vector).
- Phase 3 does NOT do cross-modal alignment.
  No pairing tables are used here.

Downstream:
A) Phase 4 (Shared latent space construction)
   - Load Phase 1 H&E backbone checkpoint -> produce H&E embeddings keyed by (sample_id, he_cell_id).
   - Load Phase 3 RA embeddings keyed by (sample_id, he_cell_id).
   - Add modality-specific adapters to map each embedding into shared_dim.
   - (Optionally) add contrastive-only projection heads (shared -> alignment).

B) Phase 5/6 (Contrastive alignment phases)
   - Use scale-matched supports:
       - L2 H&E token (anchored by he_cell_id) ↔ protein RA embedding (anchored by he_cell_id)
       - L3 H&E token ↔ Visium spot token (from Phase 2)
   - Pairing is enabled only after Phase 4 verifies distribution stability.

Key invariants to preserve
--------------------------
- Never rename: sample_id, he_cell_id, x, y for H&E anchors (Phase 0/1/3 must agree).
- Never overload he_cell_id to mean CODEX/protein cells.
  Use codex_cell_id (or prot_cell_id) for CODEX cell identities.
- Phase 3 output is RA space only; semantic alignment happens later.

"""

