"""
Phase 0: Data Contract & Project Skeleton (PyTorch Lightning)

What this phase does:
- Defines stable, explicit schemas for:
  1) Visium spot-level gene data (mtx main input; h5ad optional)
  2) H&E histology image + mock cell coordinates (he_cells)
  3) CODEX mock cell table + RA query table (queries_codex)
  4) Optional pairing tables for Level2/Level3 InfoNCE
- Builds a dataset + collate that always yields a stable batch schema
- Implements a LossRouter that automatically enables/disables InfoNCE based on data presence
- Provides a dummy LightningModule that can run N steps end-to-end

NOTE:
- This file intentionally does NOT implement real segmentation or registration.
- MockHECells and MockCODEXCellTable are treated as external providers of coordinates / features.

Dependencies:
- python >= 3.10
- pytorch
- pytorch-lightning (or lightning)
- pandas
- numpy
- scipy
- pillow (PIL)
- optionally: anndata (for .h5ad path)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

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

from PIL import Image
from scipy.io import mmread
from scipy import sparse


# ============================================================
# 0) SOFT-CODED INPUT FILE AREA (edit this section only)
# ============================================================

@dataclass
class InputPaths:
    """
    Soft-coded input paths.

    You can point to either:
    - Visium MTX bundle directory (matrix.mtx.gz + features/barcodes + tissue_positions + scalefactors)
    - Or a Visium h5ad file (optional cache)

    H&E:
    - an image file (.png or .tif)
    - a mock cell coordinate table (he_cells) in parquet/csv

    CODEX:
    - a mock codex cell table (codex_cell_table) in parquet/csv
    - a queries table (queries_codex) in parquet/csv
    - optional pairing tables (pairs_level2 / pairs_level3)
    """

    # ---- Visium ----
    visium_mtx_dir: Optional[Path] = Path("/path/to/visium_mtx_bundle_dir")  # directory containing matrix.mtx.gz etc
    visium_h5ad: Optional[Path] = None  # optional, can be used instead of mtx

    # ---- H&E ----
    he_image: Path = Path("/path/to/he_image.png")  # .png or .tif
    he_cells_table: Path = Path("/path/to/he_cells.parquet")  # MockHECells (centroids), parquet/csv

    # ---- CODEX ----
    codex_cell_table: Path = Path("/path/to/codex_cell_table.parquet")  # MockCODEXSeg (centroids + features)
    queries_codex: Path = Path("/path/to/queries_codex.parquet")  # query_id_codex + (x,y) + radius

    # ---- Optional pairings (can be None or empty) ----
    pairs_level2: Optional[Path] = None  # he_patch_id_l2 <-> query_id_codex
    pairs_level3: Optional[Path] = None  # he_patch_id_l3 <-> spot_id


# ============================================================
# 1) SOFT-CODED PARAMETER AREA (edit this section only)
# ============================================================

@dataclass
class Phase0Config:
    """
    Soft-coded parameters.

    Key principle: patch sizes and other scales are NOT hard-coded in model.
    They are parameters used to generate patch index / tensors, and can be ablated later.
    """

    # ---- Global ----
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- H&E patching ----
    # Use "patch_size_px" per level. This is the "soft encoding" of scale.
    # You can later switch to micron-based patch sizes (patch_size_um + mpp).
    patch_size_px_l1: int = 64
    patch_size_px_l2: int = 256
    patch_size_px_l3: int = 512

    # Expected channels for H&E tensors:
    # - If you load RGB PNG, C=3
    # - If you load single-channel, C=1
    he_force_rgb: bool = True

    # ---- RA (Radius-based Aggregation) ----
    # Units MUST be consistent with codex_cell_table (x,y).
    # If using pixel units, keep everything in pixel.
    # If using microns, store x,y in microns (recommended long-term).
    ra_radius_default: float = 50.0  # only used if queries table does not provide radius

    # ---- Training / data loading ----
    batch_size: int = 8
    num_workers: int = 2
    max_steps: int = 10  # Phase0 "Done": run N steps

    # ---- Dummy embedding dims (placeholders for Phase 0) ----
    d_img: int = 128
    d_gene: int = 128
    d_prot: int = 128
    d_proj: int = 128  # projection head dim for contrastive

    # ---- InfoNCE temperature (placeholder) ----
    temperature: float = 0.07

    # ---- Debug toggles ----
    print_batch_schema_every_n_steps: int = 1


# ============================================================
# 2) Utilities (seed, I/O helpers, safety checks)
# ============================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_table(path: Path) -> pd.DataFrame:
    """
    Read parquet or csv/tsv into a DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Table not found: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() in [".csv", ".tsv"]:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported table format: {path.suffix}")


def load_image(path: Path, force_rgb: bool = True) -> np.ndarray:
    """
    Load H&E image into numpy array.
    Returns: H x W x C (uint8)
    """
    img = Image.open(path)
    if force_rgb:
        img = img.convert("RGB")
    arr = np.array(img)
    # Ensure 3D (H,W,C)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def crop_patch(arr_hw_c: np.ndarray, cx: float, cy: float, patch_size: int) -> np.ndarray:
    """
    Crop a square patch centered at (cx,cy) from a HxWxC image.
    Pads with zeros if out of bounds.

    cx, cy are in pixel coordinates in the H&E image space.
    """
    H, W, C = arr_hw_c.shape
    half = patch_size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + patch_size
    y1 = y0 + patch_size

    patch = np.zeros((patch_size, patch_size, C), dtype=arr_hw_c.dtype)

    # intersection
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(W, x1), min(H, y1)

    px0, py0 = ix0 - x0, iy0 - y0
    px1, py1 = px0 + (ix1 - ix0), py0 + (iy1 - iy0)

    if ix1 > ix0 and iy1 > iy0:
        patch[py0:py1, px0:px1, :] = arr_hw_c[iy0:iy1, ix0:ix1, :]
    return patch


# ============================================================
# 3) Data Contract schemas (as documentation + minimal validators)
# ============================================================

REQUIRED_HE_CELLS_COLS = ["sample_id", "he_cell_id", "x", "y"]
REQUIRED_CODEX_CELL_COLS = ["sample_id", "codex_cell_id", "x", "y"]
REQUIRED_QUERIES_CODEX_COLS = ["sample_id", "query_id_codex", "x", "y"]  # radius optional

# Pair tables:
# pairs_level2: he_patch_id_l2 <-> query_id_codex
# pairs_level3: he_patch_id_l3 <-> spot_id
REQUIRED_PAIRS_L2_COLS = ["sample_id", "he_patch_id_l2", "query_id_codex"]
REQUIRED_PAIRS_L3_COLS = ["sample_id", "he_patch_id_l3", "spot_id"]


def validate_cols(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing required columns: {missing}\nFound: {list(df.columns)}")


# ============================================================
# 4) Visium Loader (MTX main; H5AD optional)
# ============================================================

@dataclass
class VisiumData:
    """
    Unified representation of Visium input for Phase 0.

    X_counts: sparse CSR matrix of shape [n_spots, n_genes]
    spots: DataFrame with columns:
      - sample_id
      - spot_id (barcode)
      - x, y (float; "shared coord" ideally, but for Phase 0 can be visium pixel coords)
      - in_tissue (optional bool)
      - gene_index (int index into X_counts row)
    genes: DataFrame or list of gene names (optional in Phase 0)
    """
    X_counts: sparse.csr_matrix
    spots: pd.DataFrame
    genes: Optional[pd.DataFrame] = None


def load_visium_from_mtx(visium_dir: Path, sample_id: str) -> VisiumData:
    """
    Load Visium mtx bundle.
    Expects standard 10x structure:
      - matrix.mtx.gz
      - barcodes.tsv.gz
      - features.tsv.gz
      - tissue_positions.csv (or tissue_positions_list.csv)
      - scalefactors_json.json (optional for coordinate scaling)

    Phase 0 only needs: spot_id, x, y, X_counts.
    """
    visium_dir = Path(visium_dir)
    if not visium_dir.exists():
        raise FileNotFoundError(f"visium_mtx_dir not found: {visium_dir}")

    # Find files (handle slight naming differences)
    mtx_path = None
    for cand in ["matrix.mtx.gz", "matrix.mtx"]:
        if (visium_dir / cand).exists():
            mtx_path = visium_dir / cand
            break
    if mtx_path is None:
        raise FileNotFoundError(f"Cannot find matrix.mtx(.gz) in {visium_dir}")

    barcodes_path = None
    for cand in ["barcodes.tsv.gz", "barcodes.tsv"]:
        if (visium_dir / cand).exists():
            barcodes_path = visium_dir / cand
            break
    if barcodes_path is None:
        raise FileNotFoundError(f"Cannot find barcodes.tsv(.gz) in {visium_dir}")

    features_path = None
    for cand in ["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"]:
        if (visium_dir / cand).exists():
            features_path = visium_dir / cand
            break
    if features_path is None:
        raise FileNotFoundError(f"Cannot find features.tsv(.gz) in {visium_dir}")

    # tissue positions
    tissue_pos_path = None
    for cand in ["tissue_positions.csv", "tissue_positions_list.csv"]:
        if (visium_dir / cand).exists():
            tissue_pos_path = visium_dir / cand
            break
    if tissue_pos_path is None:
        raise FileNotFoundError(f"Cannot find tissue_positions*.csv in {visium_dir}")

    # Load MTX (often genes x spots; must align to barcodes/features)
    X = mmread(str(mtx_path))
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")[0].astype(str).tolist()
    features = pd.read_csv(features_path, header=None, sep="\t")
    # 10x features.tsv: gene_id, gene_name, feature_type
    genes_df = features.rename(columns={0: "gene_id", 1: "gene_name", 2: "feature_type"}) if features.shape[1] >= 3 \
        else features.rename(columns={0: "gene_id", 1: "gene_name"})

    # Determine orientation by matching dimensions
    # If X is genes x spots -> X.shape[1] should equal len(barcodes)
    if X.shape[1] == len(barcodes) and X.shape[0] == len(genes_df):
        X_counts = X.transpose().tocsr()  # [spots, genes]
    elif X.shape[0] == len(barcodes) and X.shape[1] == len(genes_df):
        X_counts = X.tocsr()  # already [spots, genes]
    else:
        raise ValueError(
            f"MTX shape {X.shape} does not match barcodes ({len(barcodes)}) and features ({len(genes_df)})."
        )

    # tissue_positions.csv typically: barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
    tp = pd.read_csv(tissue_pos_path, header=None)
    if tp.shape[1] < 6:
        # sometimes has header row; try reading with header=0
        tp = pd.read_csv(tissue_pos_path, header=0)
        # then we assume columns exist
        # For Phase 0 we only need barcode + pixel coordinates; handle flexibly
        # If your file has standard column names, map them here as needed.

    # Try to parse standard no-header format first
    if tp.shape[1] >= 6 and tp.columns.tolist() == list(range(tp.shape[1])):
        tp.columns = ["spot_id", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"] + \
                     [f"extra_{i}" for i in range(6, tp.shape[1])]

    # Keep only barcodes present in matrix
    tp["spot_id"] = tp["spot_id"].astype(str)
    bset = set(barcodes)
    tp = tp[tp["spot_id"].isin(bset)].copy()

    # Build spots table aligned with X_counts row order (barcodes order)
    spot_to_index = {b: i for i, b in enumerate(barcodes)}
    tp["gene_index"] = tp["spot_id"].map(spot_to_index).astype(int)
    tp["sample_id"] = sample_id

    # For Phase 0, use pixel coords as x/y.
    # IMPORTANT: later you may map to a shared coordinate system via registration.
    # Here we choose x = pxl_col, y = pxl_row (common convention).
    if "pxl_col" in tp.columns and "pxl_row" in tp.columns:
        tp["x"] = tp["pxl_col"].astype(float)
        tp["y"] = tp["pxl_row"].astype(float)
    else:
        # fallback: try columns that might exist
        raise ValueError("tissue_positions file missing pxl_row/pxl_col fields; please map columns accordingly.")

    # Keep minimal columns
    spots = tp[["sample_id", "spot_id", "x", "y", "gene_index"] + (["in_tissue"] if "in_tissue" in tp.columns else [])].copy()

    return VisiumData(X_counts=X_counts, spots=spots, genes=genes_df)


def load_visium_from_h5ad(h5ad_path: Path, sample_id: str) -> VisiumData:
    """
    Optional: load Visium from anndata (.h5ad).
    This is a CACHE path, not the authoritative raw counts path in Phase 0.

    Requirements:
    - adata.X is counts or can be used as gene matrix
    - adata.obs contains spot barcodes (e.g., obs_names) and x/y columns
    """
    try:
        import anndata as ad
    except ImportError as e:
        raise ImportError("anndata is required to load .h5ad. Install with `pip install anndata`.") from e

    adata = ad.read_h5ad(h5ad_path)
    X = adata.X
    if sparse.issparse(X):
        X_counts = X.tocsr()
    else:
        X_counts = sparse.csr_matrix(X)

    # spot_id
    spot_ids = adata.obs_names.astype(str).tolist()

    # x/y columns - adapt as needed for your adata convention
    # common candidates: "x", "y", "pxl_col", "pxl_row", "imagecol", "imagerow"
    obs = adata.obs.copy()
    obs["spot_id"] = spot_ids

    # Find x/y columns
    xcol = None
    ycol = None
    for candx, candy in [("x", "y"), ("pxl_col", "pxl_row"), ("imagecol", "imagerow")]:
        if candx in obs.columns and candy in obs.columns:
            xcol, ycol = candx, candy
            break
    if xcol is None:
        raise ValueError("Cannot find x/y columns in adata.obs. Please map x/y columns in loader.")

    spots = pd.DataFrame({
        "sample_id": sample_id,
        "spot_id": obs["spot_id"].astype(str).values,
        "x": obs[xcol].astype(float).values,
        "y": obs[ycol].astype(float).values,
        "gene_index": np.arange(len(spot_ids), dtype=int),
    })

    genes_df = None
    if adata.var is not None:
        genes_df = adata.var.copy()
        genes_df["gene_name"] = genes_df.index.astype(str)

    return VisiumData(X_counts=X_counts, spots=spots, genes=genes_df)


# ============================================================
# 5) H&E Patch Index builder (from he_cells + patch sizes)
# ============================================================

def build_he_patches_index(
    he_cells: pd.DataFrame,
    cfg: Phase0Config,
) -> pd.DataFrame:
    """
    Build a patch index table from he_cells (MockHECells).
    This table is the "contract" that defines patch centers and sizes.

    Output columns (minimal):
    - sample_id
    - he_cell_id
    - level (1/2/3)
    - he_patch_id_l{level}
    - center_x, center_y
    - patch_size_px

    Note:
    - We do NOT store image pixels in this index.
    - This allows lazy patch generation during training.
    """
    validate_cols(he_cells, REQUIRED_HE_CELLS_COLS, "he_cells")

    rows = []
    for _, r in he_cells.iterrows():
        sample_id = str(r["sample_id"])
        he_cell_id = str(r["he_cell_id"])
        cx = float(r["x"])
        cy = float(r["y"])

        for level, patch_size in [
            (1, cfg.patch_size_px_l1),
            (2, cfg.patch_size_px_l2),
            (3, cfg.patch_size_px_l3),
        ]:
            patch_id = f"{sample_id}__{he_cell_id}__L{level}"
            rows.append({
                "sample_id": sample_id,
                "he_cell_id": he_cell_id,
                "level": level,
                f"he_patch_id_l{level}": patch_id,
                "center_x": cx,
                "center_y": cy,
                "patch_size_px": int(patch_size),
            })

    he_patches_index = pd.DataFrame(rows)

    # Optional: for convenience, we can keep a unified patch_id column too
    he_patches_index["he_patch_id"] = he_patches_index.apply(
        lambda x: x[f"he_patch_id_l{int(x['level'])}"], axis=1
    )
    return he_patches_index


# ============================================================
# 6) RA Aggregation (CODEX cell table + queries)
# ============================================================

def radius_aggregate_protein(
    codex_cell_table: pd.DataFrame,
    queries: pd.DataFrame,
    feature_cols: List[str],
    default_radius: float,
) -> pd.DataFrame:
    """
    Compute Radius-based Aggregation (RA) features.

    Inputs:
    - codex_cell_table: rows are cells, with x,y and protein feature columns
    - queries: rows are query points, with x,y and optional radius
    - feature_cols: which columns in codex_cell_table are protein features
    - default_radius: used if queries doesn't have 'radius'

    Output:
    - ra_table: one row per query_id_codex with aggregated protein vector
      columns: sample_id, query_id_codex, agg_{feature}, n_cells_in_radius
    """
    validate_cols(codex_cell_table, REQUIRED_CODEX_CELL_COLS, "codex_cell_table")
    validate_cols(queries, REQUIRED_QUERIES_CODEX_COLS, "queries_codex")

    # Ensure radius column exists
    if "radius" not in queries.columns:
        queries = queries.copy()
        queries["radius"] = float(default_radius)

    # Pre-extract coords for speed
    # NOTE: This is a naive O(NQ * NC) implementation, good enough for Phase 0.
    # Later you should use KD-tree / ball tree for speed.
    out_rows = []
    for _, q in queries.iterrows():
        sid = str(q["sample_id"])
        qid = str(q["query_id_codex"])
        qx, qy = float(q["x"]), float(q["y"])
        rad = float(q["radius"])

        sub_cells = codex_cell_table[codex_cell_table["sample_id"].astype(str) == sid]
        if len(sub_cells) == 0:
            agg = np.zeros(len(feature_cols), dtype=np.float32)
            n = 0
        else:
            dx = sub_cells["x"].astype(float).values - qx
            dy = sub_cells["y"].astype(float).values - qy
            dist2 = dx * dx + dy * dy
            mask = dist2 <= (rad * rad)
            chosen = sub_cells.loc[mask, feature_cols]
            n = chosen.shape[0]
            if n == 0:
                agg = np.zeros(len(feature_cols), dtype=np.float32)
            else:
                # mean aggregation (can switch to sum/median/etc later)
                agg = chosen.astype(float).mean(axis=0).values.astype(np.float32)

        row = {"sample_id": sid, "query_id_codex": qid, "n_cells_in_radius": int(n)}
        for i, f in enumerate(feature_cols):
            row[f"agg_{f}"] = float(agg[i])
        out_rows.append(row)

    return pd.DataFrame(out_rows)


# ============================================================
# 7) Dataset & Collate (stable batch schema)
# ============================================================

class MultiModalPhase0Dataset(Dataset):
    """
    A Phase 0 dataset that yields per-item references:
    - H&E patch center + size (from he_patches_index)
    - Visium spot row index (optional; can be missing)
    - CODEX RA query id (optional; can be missing)

    Key design:
    - Pairings are optional.
    - The dataset produces an item dict with stable keys.
    - Collate will assemble to stable batch schema.

    Sampling strategy in Phase 0:
    - We iterate over he_cells (each defines one cell i), and derive:
      - P1/P2/P3 patch IDs
      - optional L2 pair: P2 <-> query_id_codex
      - optional L3 pair: P3 <-> spot_id

    If pair tables are missing or empty, we still yield items, but without pair ids.
    """

    def __init__(
        self,
        he_image_arr: np.ndarray,
        he_cells: pd.DataFrame,
        he_patches_index: pd.DataFrame,
        visium: Optional[VisiumData],
        codex_cell_table: pd.DataFrame,
        ra_table: pd.DataFrame,
        pairs_l2: Optional[pd.DataFrame],
        pairs_l3: Optional[pd.DataFrame],
        cfg: Phase0Config,
    ) -> None:
        super().__init__()
        self.he_img = he_image_arr  # H x W x C
        self.he_cells = he_cells.reset_index(drop=True)
        self.he_patches_index = he_patches_index
        self.visium = visium
        self.codex_cell_table = codex_cell_table
        self.ra_table = ra_table
        self.pairs_l2 = pairs_l2
        self.pairs_l3 = pairs_l3
        self.cfg = cfg

        validate_cols(self.he_cells, REQUIRED_HE_CELLS_COLS, "he_cells")

        # Build quick lookups for patch index
        # he_patch_id is unique per (sample_id, he_cell_id, level)
        self.patch_lookup = {}
        for _, r in self.he_patches_index.iterrows():
            self.patch_lookup[(str(r["sample_id"]), str(r["he_cell_id"]), int(r["level"]))] = {
                "patch_id": str(r["he_patch_id"]),
                "center_x": float(r["center_x"]),
                "center_y": float(r["center_y"]),
                "patch_size_px": int(r["patch_size_px"]),
            }

        # Build RA lookup: query_id_codex -> aggregated vector
        # RA table columns: agg_{feature}
        self.ra_lookup = {}
        ra_feat_cols = [c for c in self.ra_table.columns if c.startswith("agg_")]
        for _, r in self.ra_table.iterrows():
            sid = str(r["sample_id"])
            qid = str(r["query_id_codex"])
            vec = r[ra_feat_cols].astype(float).values.astype(np.float32)
            self.ra_lookup[(sid, qid)] = vec

        # Pairing lookups (optional)
        self.l2_lookup = {}
        if self.pairs_l2 is not None and len(self.pairs_l2) > 0:
            validate_cols(self.pairs_l2, REQUIRED_PAIRS_L2_COLS, "pairs_level2")
            for _, r in self.pairs_l2.iterrows():
                sid = str(r["sample_id"])
                pid = str(r["he_patch_id_l2"])
                qid = str(r["query_id_codex"])
                self.l2_lookup[(sid, pid)] = qid

        self.l3_lookup = {}
        if self.pairs_l3 is not None and len(self.pairs_l3) > 0:
            validate_cols(self.pairs_l3, REQUIRED_PAIRS_L3_COLS, "pairs_level3")
            for _, r in self.pairs_l3.iterrows():
                sid = str(r["sample_id"])
                pid = str(r["he_patch_id_l3"])
                spid = str(r["spot_id"])
                self.l3_lookup[(sid, pid)] = spid

        # Visium lookup: spot_id -> gene_index (row in X_counts)
        self.spot_to_gene_index = {}
        if self.visium is not None:
            for _, r in self.visium.spots.iterrows():
                self.spot_to_gene_index[(str(r["sample_id"]), str(r["spot_id"]))] = int(r["gene_index"])

    def __len__(self) -> int:
        return len(self.he_cells)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.he_cells.iloc[idx]
        sid = str(r["sample_id"])
        he_cell_id = str(r["he_cell_id"])
        cx = float(r["x"])
        cy = float(r["y"])

        # Build patch refs for three levels
        patch_refs = {}
        patch_tensors = {}
        for level in [1, 2, 3]:
            info = self.patch_lookup[(sid, he_cell_id, level)]
            patch_id = info["patch_id"]
            ps = info["patch_size_px"]
            px = crop_patch(self.he_img, info["center_x"], info["center_y"], ps)  # HxWxC uint8

            # Convert to torch tensor [C,H,W], float32 in [0,1]
            t = torch.from_numpy(px).permute(2, 0, 1).float() / 255.0
            patch_refs[f"patch_id_l{level}"] = patch_id
            patch_refs[f"patch_size_l{level}"] = ps
            patch_tensors[f"patch_l{level}"] = t

        # Optional: Level2 pairing -> query_id_codex
        qid = None
        if "patch_id_l2" in patch_refs:
            qid = self.l2_lookup.get((sid, patch_refs["patch_id_l2"]), None)

        ra_vec = None
        if qid is not None:
            ra_vec = self.ra_lookup.get((sid, qid), None)  # np array
            if ra_vec is not None:
                ra_vec = torch.from_numpy(ra_vec).float()

        # Optional: Level3 pairing -> spot_id -> gene vector
        spot_id = None
        gene_vec = None
        if "patch_id_l3" in patch_refs and self.visium is not None:
            spot_id = self.l3_lookup.get((sid, patch_refs["patch_id_l3"]), None)
            if spot_id is not None:
                gidx = self.spot_to_gene_index.get((sid, spot_id), None)
                if gidx is not None:
                    # Extract sparse row -> dense vector (Phase 0 simplicity)
                    row = self.visium.X_counts.getrow(gidx)
                    gene_vec = torch.from_numpy(row.toarray().squeeze(0).astype(np.float32))

        item = {
            "sample_id": sid,
            "he_cell_id": he_cell_id,
            "center_xy": torch.tensor([cx, cy], dtype=torch.float32),

            # H&E patches (always present)
            "he_patches": patch_tensors,  # dict: patch_l1, patch_l2, patch_l3
            "he_patch_meta": patch_refs,  # dict with patch ids/sizes

            # Optional: Protein side (RA)
            "query_id_codex": qid,         # may be None
            "ra_protein_vec": ra_vec,      # may be None

            # Optional: Gene side (Visium)
            "spot_id": spot_id,            # may be None
            "gene_vec": gene_vec,          # may be None
        }
        return item


def collate_phase0(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate into a stable batch schema.

    IMPORTANT: Some fields may be missing (None). We keep them as None or build masks.
    This enables LossRouter to decide which losses to compute.
    """
    # Always-present fields
    sample_ids = [b["sample_id"] for b in batch]
    he_cell_ids = [b["he_cell_id"] for b in batch]
    centers = torch.stack([b["center_xy"] for b in batch], dim=0)  # [B,2]

    # H&E patches: stack each level
    he_l1 = torch.stack([b["he_patches"]["patch_l1"] for b in batch], dim=0)  # [B,C,H,W]
    he_l2 = torch.stack([b["he_patches"]["patch_l2"] for b in batch], dim=0)
    he_l3 = torch.stack([b["he_patches"]["patch_l3"] for b in batch], dim=0)

    # Protein RA vectors (optional)
    ra_list = [b["ra_protein_vec"] for b in batch]
    ra_mask = torch.tensor([v is not None for v in ra_list], dtype=torch.bool)
    ra_vec = None
    if ra_mask.any():
        # pad missing with zeros to keep tensor shape stable
        dim = ra_list[ra_mask.nonzero(as_tuple=True)[0][0]].shape[0]
        ra_vec = torch.zeros((len(batch), dim), dtype=torch.float32)
        for i, v in enumerate(ra_list):
            if v is not None:
                ra_vec[i] = v

    # Gene vectors (optional)
    gene_list = [b["gene_vec"] for b in batch]
    gene_mask = torch.tensor([v is not None for v in gene_list], dtype=torch.bool)
    gene_vec = None
    if gene_mask.any():
        dim = gene_list[gene_mask.nonzero(as_tuple=True)[0][0]].shape[0]
        gene_vec = torch.zeros((len(batch), dim), dtype=torch.float32)
        for i, v in enumerate(gene_list):
            if v is not None:
                gene_vec[i] = v

    # Patch ids + pairing keys (optional metadata)
    patch_id_l2 = [b["he_patch_meta"]["patch_id_l2"] for b in batch]
    patch_id_l3 = [b["he_patch_meta"]["patch_id_l3"] for b in batch]
    query_id_codex = [b["query_id_codex"] for b in batch]
    spot_id = [b["spot_id"] for b in batch]

    out = {
        # identifiers
        "sample_id": sample_ids,
        "he_cell_id": he_cell_ids,
        "center_xy": centers,

        # H&E patch tensors (always)
        "he_l1": he_l1,
        "he_l2": he_l2,
        "he_l3": he_l3,

        # optional modalities + masks
        "ra_protein_vec": ra_vec,   # [B, P] or None
        "ra_mask": ra_mask,         # [B]
        "gene_vec": gene_vec,       # [B, G] or None
        "gene_mask": gene_mask,     # [B]

        # optional IDs (for debugging / later richer pairing)
        "he_patch_id_l2": patch_id_l2,
        "he_patch_id_l3": patch_id_l3,
        "query_id_codex": query_id_codex,
        "spot_id": spot_id,
    }
    return out


# ============================================================
# 8) Dummy encoders + LossRouter (Phase 0 placeholders)
# ============================================================

class DummyImageEncoder(nn.Module):
    """
    Placeholder for ViT backbone.
    Phase 0 goal: accept [B,C,H,W] and output [B,d_img].
    """
    def __init__(self, d_out: int):
        super().__init__()
        # Simple conv + pool as placeholder
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DummyMLP(nn.Module):
    """
    Placeholder for gene/protein encoders.
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    CLIP-style symmetric InfoNCE (simplified).
    z1, z2: [B, D]
    """
    z1 = nn.functional.normalize(z1, dim=-1)
    z2 = nn.functional.normalize(z2, dim=-1)

    logits = (z1 @ z2.t()) / temperature  # [B,B]
    labels = torch.arange(z1.shape[0], device=z1.device)
    loss_12 = nn.functional.cross_entropy(logits, labels)
    loss_21 = nn.functional.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_12 + loss_21)


class LossRouter(nn.Module):
    """
    Decides which losses to compute based on batch content.

    Phase 0 behavior:
    - Always can compute "placeholder self-supervised" losses if desired (set to 0 here)
    - Compute Level2 InfoNCE only if ra_protein_vec exists for some samples
    - Compute Level3 InfoNCE only if gene_vec exists for some samples
    """
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        batch: Dict[str, Any],
        z_l2: torch.Tensor,
        z_l3: torch.Tensor,
        prot_z: Optional[torch.Tensor],
        gene_z: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        # --- Level2 H&E <-> Protein InfoNCE (only where RA exists) ---
        if prot_z is not None and batch["ra_mask"].any():
            mask = batch["ra_mask"]
            losses["loss_infonce_l2"] = info_nce_loss(z_l2[mask], prot_z[mask], self.temperature)
        else:
            losses["loss_infonce_l2"] = torch.tensor(0.0, device=z_l2.device)

        # --- Level3 H&E <-> Gene InfoNCE (only where gene exists) ---
        if gene_z is not None and batch["gene_mask"].any():
            mask = batch["gene_mask"]
            losses["loss_infonce_l3"] = info_nce_loss(z_l3[mask], gene_z[mask], self.temperature)
        else:
            losses["loss_infonce_l3"] = torch.tensor(0.0, device=z_l3.device)

        # Placeholder: DINO / self-supervised losses can be integrated later
        losses["loss_dino_l1"] = torch.tensor(0.0, device=z_l2.device)
        losses["loss_dino_l2"] = torch.tensor(0.0, device=z_l2.device)
        losses["loss_dino_l3"] = torch.tensor(0.0, device=z_l2.device)

        # Total (Phase 0): just sum; later add weights/schedules
        losses["loss_total"] = losses["loss_infonce_l2"] + losses["loss_infonce_l3"] + \
                               losses["loss_dino_l1"] + losses["loss_dino_l2"] + losses["loss_dino_l3"]
        return losses


# ============================================================
# 9) LightningModule (dummy but runnable end-to-end)
# ============================================================

class Phase0LightningModule(L.LightningModule):
    """
    A runnable skeleton that consumes the Phase 0 batch schema.
    Replace DummyImageEncoder with ViT later; replace DummyMLP with real encoders later.
    """
    def __init__(self, cfg: Phase0Config, gene_dim: int, prot_dim: int):
        super().__init__()
        self.cfg = cfg

        # --- Encoders ---
        self.img_encoder = DummyImageEncoder(cfg.d_img)
        self.prot_encoder = DummyMLP(prot_dim, cfg.d_prot)
        self.gene_encoder = DummyMLP(gene_dim, cfg.d_gene)

        # --- Project heads to contrastive space ---
        self.proj_l2 = nn.Linear(cfg.d_img, cfg.d_proj)
        self.proj_l3 = nn.Linear(cfg.d_img, cfg.d_proj)
        self.proj_prot = nn.Linear(cfg.d_prot, cfg.d_proj)
        self.proj_gene = nn.Linear(cfg.d_gene, cfg.d_proj)

        self.loss_router = LossRouter(cfg.temperature)

        self._step_count = 0

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        # Print batch schema periodically (Phase 0 debug requirement)
        if self.cfg.print_batch_schema_every_n_steps > 0 and (self._step_count % self.cfg.print_batch_schema_every_n_steps == 0):
            self._print_batch_schema(batch)

        # H&E encodings
        z_l2_img = self.img_encoder(batch["he_l2"])
        z_l3_img = self.img_encoder(batch["he_l3"])

        z_l2 = self.proj_l2(z_l2_img)
        z_l3 = self.proj_l3(z_l3_img)

        # Protein encoding (optional)
        prot_z = None
        if batch["ra_protein_vec"] is not None:
            prot_h = self.prot_encoder(batch["ra_protein_vec"])
            prot_z = self.proj_prot(prot_h)

        # Gene encoding (optional)
        gene_z = None
        if batch["gene_vec"] is not None:
            gene_h = self.gene_encoder(batch["gene_vec"])
            gene_z = self.proj_gene(gene_h)

        losses = self.loss_router(batch, z_l2, z_l3, prot_z, gene_z)
        self.log_dict({k: v.detach() for k, v in losses.items()}, prog_bar=True)

        self._step_count += 1
        return losses["loss_total"]

    def _print_batch_schema(self, batch: Dict[str, Any]) -> None:
        """
        Debug-friendly batch schema print.
        """
        def shape(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return list(x.shape)
            return type(x).__name__

        msg = {
            "he_l1": shape(batch["he_l1"]),
            "he_l2": shape(batch["he_l2"]),
            "he_l3": shape(batch["he_l3"]),
            "ra_protein_vec": shape(batch["ra_protein_vec"]),
            "ra_mask": shape(batch["ra_mask"]),
            "gene_vec": shape(batch["gene_vec"]),
            "gene_mask": shape(batch["gene_mask"]),
            "he_patch_id_l2": "list[str]",
            "he_patch_id_l3": "list[str]",
            "query_id_codex": "list[optional str]",
            "spot_id": "list[optional str]",
        }
        print("[Phase0] Batch schema:", msg)


# ============================================================
# 10) DataModule (wires everything together)
# ============================================================

class Phase0DataModule(L.LightningDataModule):
    def __init__(self, paths: InputPaths, cfg: Phase0Config, sample_id: str):
        super().__init__()
        self.paths = paths
        self.cfg = cfg
        self.sample_id = sample_id

        # These will be populated in setup()
        self.dataset: Optional[MultiModalPhase0Dataset] = None
        self.gene_dim: Optional[int] = None
        self.prot_dim: Optional[int] = None

    def setup(self, stage: Optional[str] = None):
        set_seed(self.cfg.seed)

        # ---- Load H&E image ----
        he_img = load_image(self.paths.he_image, force_rgb=self.cfg.he_force_rgb)

        # ---- Load MockHECells ----
        he_cells = read_table(self.paths.he_cells_table)
        validate_cols(he_cells, REQUIRED_HE_CELLS_COLS, "he_cells")

        # ---- Build he_patches_index (contract table for patch extraction) ----
        he_patches_index = build_he_patches_index(he_cells, self.cfg)

        # ---- Load Visium ----
        visium = None
        if self.paths.visium_h5ad is not None:
            visium = load_visium_from_h5ad(self.paths.visium_h5ad, sample_id=self.sample_id)
        elif self.paths.visium_mtx_dir is not None:
            visium = load_visium_from_mtx(self.paths.visium_mtx_dir, sample_id=self.sample_id)

        # Determine gene_dim (if visium exists)
        if visium is not None:
            self.gene_dim = int(visium.X_counts.shape[1])
        else:
            # Phase 0 still runnable without gene
            self.gene_dim = 16  # dummy fallback; gene branch will be skipped if gene_vec is None

        # ---- Load MockCODEX cell table ----
        codex_cell_table = read_table(self.paths.codex_cell_table)
        validate_cols(codex_cell_table, REQUIRED_CODEX_CELL_COLS, "codex_cell_table")

        # Determine which columns are protein features.
        # Convention: any column not in (sample_id,codex_cell_id,x,y,...) is a feature.
        non_feature = set(["sample_id", "codex_cell_id", "x", "y"])
        feature_cols = [c for c in codex_cell_table.columns if c not in non_feature]
        if len(feature_cols) == 0:
            raise ValueError("codex_cell_table has no protein feature columns. Add marker columns or a feature vector expansion.")
        self.prot_dim = len(feature_cols)

        # ---- Load queries_codex ----
        queries_codex = read_table(self.paths.queries_codex)
        validate_cols(queries_codex, REQUIRED_QUERIES_CODEX_COLS, "queries_codex")

        # ---- Compute RA table (Phase 0 naive implementation) ----
        ra_table = radius_aggregate_protein(
            codex_cell_table=codex_cell_table,
            queries=queries_codex,
            feature_cols=feature_cols,
            default_radius=self.cfg.ra_radius_default,
        )

        # ---- Optional pair tables ----
        pairs_l2 = read_table(self.paths.pairs_level2) if self.paths.pairs_level2 is not None else None
        pairs_l3 = read_table(self.paths.pairs_level3) if self.paths.pairs_level3 is not None else None

        # ---- Construct dataset ----
        self.dataset = MultiModalPhase0Dataset(
            he_image_arr=he_img,
            he_cells=he_cells,
            he_patches_index=he_patches_index,
            visium=visium,
            codex_cell_table=codex_cell_table,
            ra_table=ra_table,
            pairs_l2=pairs_l2,
            pairs_l3=pairs_l3,
            cfg=self.cfg,
        )

    def train_dataloader(self):
        assert self.dataset is not None
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_phase0,
            pin_memory=True,
        )


# ============================================================
# 11) Main entry (Phase 0 runnable demo)
# ============================================================

def main():
    # --------------------------
    # EDIT HERE: define sample_id and paths
    # --------------------------
    sample_id = "HTA12_269"  # change to your sample
    paths = InputPaths(
        visium_mtx_dir=Path("/coh_labs/dits/bzhang/multi-omics/hta12_269/hta12_269_2_visium"),
        visium_h5ad=None,

        he_image=Path("/path/to/hta12_269_he.png"),
        he_cells_table=Path("/path/to/he_cells.parquet"),

        codex_cell_table=Path("/path/to/codex_cell_table.parquet"),
        queries_codex=Path("/path/to/queries_codex.parquet"),

        pairs_level2=None,  # can be set later
        pairs_level3=None,  # can be set later
    )

    cfg = Phase0Config(
        batch_size=4,
        max_steps=10,
        num_workers=0,  # set >0 on real runs
    )

    dm = Phase0DataModule(paths=paths, cfg=cfg, sample_id=sample_id)
    dm.setup()

    # Determine dims for module
    gene_dim = dm.gene_dim if dm.gene_dim is not None else 16
    prot_dim = dm.prot_dim if dm.prot_dim is not None else 16

    model = Phase0LightningModule(cfg=cfg, gene_dim=gene_dim, prot_dim=prot_dim)

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=cfg.max_steps,
        log_every_n_steps=1,
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()


# ============================================================
# 12) DOWNSTREAM USAGE NOTES (Phase 0 -> Phase 1-7)
# ============================================================

"""
Downstream usage notes (how Phase 0 artifacts are consumed later):

A) he_cells (MockHECells)
- Phase 0: provides cell coordinate (x_i,y_i) to build patch index and queries.
- Phase 1+: can be replaced by real H&E segmentation output (centroids and optionally masks).
- Must keep schema stable: sample_id, he_cell_id, x, y.

B) he_patches_index
- Phase 0: built from he_cells + patch_size per level (soft-coded).
- Phase 1+: allows patch size ablation without changing training loop.
- Always provides: level, patch_size_px, center_x, center_y, he_patch_id.

C) visium input (mtx bundle or h5ad)
- Phase 0: loader produces VisiumData with X_counts and spots table.
- Phase 1+: you can add preprocessing (normalize/log1p/HVG) but keep a "raw counts path" for reproducibility.
- You can also add neighborhood graphs (KNN local/global) later; do NOT mix into Phase 0 contract.

D) codex_cell_table (MockCODEXSeg)
- Phase 0: provides centroids + protein features for aggregation.
- Phase 2+: can be replaced by real CODEX segmentation output.
- Must keep schema stable: sample_id, codex_cell_id, x, y, feature columns.

E) queries_codex
- Phase 0: defines RA centers (x,y) and radius.
- Generated from he_cells projection OR other strategies. This is intentionally decoupled from segmentation quality.
- Must keep schema stable: sample_id, query_id_codex, x, y, radius (optional but recommended).

F) pairs_level2 / pairs_level3 (optional supervision plug-ins)
- Phase 0: can be None/empty. LossRouter must skip InfoNCE gracefully.
- Phase 5+: once registration improves, provide real pairing tables to enable InfoNCE.
- Keeping these as external tables enforces "registration-agnostic" training loop.

G) Replace Dummy encoders with real ones
- Replace DummyImageEncoder -> ViT backbone (DINO)
- Replace DummyMLP -> Gene/Protein encoders
- Do NOT change dataset/batch schema; only replace model modules.
"""

