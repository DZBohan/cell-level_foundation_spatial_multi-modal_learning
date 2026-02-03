# phase1_he_l1_dino_closed_loop.py
"""
Phase 1: H&E single-modality closed loop (train Level-1 DINO only)
==================================================================

Phase 1 goal in your 8-phase plan:
- Run self-supervised learning on H&E at cell-scale (L1) only (DINO: student-teacher + EMA).
- Produce a stable H&E backbone checkpoint that can be reused downstream.
- Keep full alignment with Phase 0 contracts:
  - Patch index / patch ids follow the same naming and generation logic as Phase 0 (he_patches_index).
  - L2/L3 patches can still be forwarded to representations (for sanity/compatibility) but do NOT contribute
    to any loss and do NOT receive gradients.
  - Downstream Phase 5/6 InfoNCE alignment will use L2/L3 where "scale semantics" are defined by cross-modal
    alignment, not by DINO at this stage.

Important notes (consistent with our earlier discussion):
- Phase 1 does NOT require registration / pairing / protein / gene.
- Phase 1 L1 DINO loss is strictly self-supervised: two augmented views of the same L1 patch.
- Phase 1 can use real segmentation outputs (mask/centroid) or the Phase 0 mock centroids (he_cells_table).
- Engineering best practice is to move cropping + augmentation to a more efficient place (WSI tiling / OpenSlide / cache),
  but for a first closed-loop validation, this script provides:
  - a lightweight numpy/PIL crop for small-image demos and sanity checks;
  - a placeholder interface for WSI tile readers (TODO).

Dependencies:
- python >= 3.10
- torch
- lightning (or pytorch_lightning)
- numpy, pandas
- pillow
- (optional) timm: if you want a more standard ViT backbone (this script includes a lightweight fallback backbone)

"""

from __future__ import annotations

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

# Lightning import (works for both "lightning" and "pytorch_lightning")
try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L

from PIL import Image


# ============================================================
# 0) SOFT-CODED INPUT AREA (edit only here)
# ============================================================

@dataclass
class Phase1InputPaths:
    """
    Soft-coded input paths for Phase 1.

    Phase 0 handshake / interface notes:
    - he_image: same concept as Phase 0 (he_image). We keep the naming consistent.
    - he_cells_table: Phase 0 contract table (sample_id, he_cell_id, x, y).
      - Phase 1 can reuse it directly.
      - Even if you later replace it with real segmentation-derived centroids, keep the schema stable.
    - he_patches_index (optional):
      - If you already saved he_patches_index in Phase 0, you can load it here and reuse.
      - If you did NOT save it (Phase 0 only built it in memory), Phase 1 can rebuild it from
        he_cells_table + cfg.patch_size_l1.
        This does NOT break the contract because the contract is defined by schema + deterministic rules.

    """

    sample_id: str = "HTA12_269"

    # --- H&E image ---
    he_image: Path = Path("/path/to/he_image.png")  # PNG/TIF are both fine (small-image demo)

    # --- he_cells_table: Phase 0 contract ---
    he_cells_table: Path = Path("/path/to/he_cells.parquet")  # columns: sample_id, he_cell_id, x, y

    # --- Optional: he_patches_index prebuilt in Phase 0 ---
    # If None, we build L1 patch index from he_cells_table.
    he_patches_index: Optional[Path] = None  # parquet/csv


@dataclass
class Phase1Config:
    """
    Soft-coded training and data parameters for Phase 1.

    Notes:
    - Phase 1 trains only L1 DINO, so L1 patch size is the only scale required for the loss.
    - However, to keep Phase 0 / downstream interface compatibility, we can still generate and forward L2/L3
      (without loss and without gradients).

    """

    # ---- global ----
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- patch sizes ----
    # L1 is the training scale for Phase 1 (cell-scale)
    patch_size_px_l1: int = 64

    # L2/L3 are not trained in Phase 1, but can be forwarded for representation sanity checks (optional)
    patch_size_px_l2: int = 256
    patch_size_px_l3: int = 512
    forward_l2_l3_for_sanity: bool = True  # True: forward them; False: process only L1

    # ---- image loading ----
    he_force_rgb: bool = True  # DINO typically assumes RGB inputs

    # ---- dataloader ----
    batch_size: int = 64
    num_workers: int = 4

    # ---- training schedule ----
    max_steps: int = 2000
    lr: float = 1e-4
    weight_decay: float = 0.04
    warmup_steps: int = 200

    # ---- AMP / accumulation ----
    use_amp: bool = True
    grad_accum_steps: int = 1

    # ---- DINO hyperparams ----
    # backbone / embedding dimension
    embed_dim: int = 384   # 384 (ViT-S) is common; you can switch to 768 etc.
    proj_dim: int = 1024   # DINO head hidden dim
    out_dim: int = 65536   # DINO output dimension (original DINO often uses 65536)
    num_prototypes: int = 0  # optional: prototype head (disabled by default)

    # teacher EMA
    teacher_momentum: float = 0.996       # base EMA momentum
    teacher_momentum_final: float = 1.0   # cosine schedule to 1.0

    # DINO temperatures
    student_temp: float = 0.1
    teacher_temp: float = 0.04
    teacher_temp_warmup_steps: int = 500
    teacher_temp_warmup_start: float = 0.04

    # DINO centering
    center_momentum: float = 0.9

    # ---- debug / logging ----
    print_every_n_steps: int = 50
    log_embedding_stats_every_n_steps: int = 50

    # ---- output ----
    out_dir: Path = Path("./phase1_outputs")
    save_embeddings_sanity: bool = True
    save_embeddings_every_n_steps: int = 500  # save small embedding snapshots for sanity checks (optional)


# ============================================================
# 1) Utilities: seeding / I/O / cropping
# ============================================================

REQUIRED_HE_CELLS_COLS = ["sample_id", "he_cell_id", "x", "y"]


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


def load_image(path: Path, force_rgb: bool = True) -> np.ndarray:
    """
    Phase 1 demo reader: load the entire H&E image into numpy.

    This is acceptable for:
    - small images;
    - demo/sanity runs.

    For real WSI training you should:
    - read from WSI via OpenSlide or a tile reader;
    - cache tiles and avoid copying the full image into each DataLoader worker;
    - move expensive augmentation into a more efficient pipeline.

    """
    img = Image.open(path)
    if force_rgb:
        img = img.convert("RGB")
    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def crop_patch(arr_hw_c: np.ndarray, cx: float, cy: float, patch_size: int) -> np.ndarray:
    """
    Crop a square patch centered at (cx, cy) from a HxWxC image; pad with zeros if out of bounds.

    Coordinate system requirement:
    - (cx, cy) must be in the same coordinate system as the loaded he_image.
      If he_cells_table stores full-resolution pixel coordinates, he_image must be full-resolution.
      If you use a downsampled image, you must rescale coordinates accordingly.
    - Long-term recommendation: store coordinates in microns and keep a consistent MPP.

    """
    H, W, C = arr_hw_c.shape
    half = patch_size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + patch_size
    y1 = y0 + patch_size

    patch = np.zeros((patch_size, patch_size, C), dtype=arr_hw_c.dtype)

    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(W, x1), min(H, y1)

    px0, py0 = ix0 - x0, iy0 - y0
    px1, py1 = px0 + (ix1 - ix0), py0 + (iy1 - iy0)

    if ix1 > ix0 and iy1 > iy0:
        patch[py0:py1, px0:px1, :] = arr_hw_c[iy0:iy1, ix0:ix1, :]
    return patch


# ============================================================
# 2) Phase 0 contract reuse: he_patches_index (L1/L2/L3)
# ============================================================

def build_he_patches_index_from_cells(
    he_cells: pd.DataFrame,
    cfg: Phase1Config,
) -> pd.DataFrame:
    """
    Phase 0 he_patches_index generation logic (same schema, same naming philosophy).

    If Phase 1 is not provided with an on-disk he_patches_index file, we rebuild it from:
    - he_cells_table
    - cfg.patch_size_px_l1/l2/l3

    This does not break the contract because the contract is defined by:
    - stable schema;
    - deterministic generation rules.

    Output columns (minimal):
    - sample_id, he_cell_id, level, he_patch_id, center_x, center_y, patch_size_px
    - and he_patch_id_l{level} (consistent with Phase 0 naming)

    """
    validate_cols(he_cells, REQUIRED_HE_CELLS_COLS, "he_cells")

    rows = []
    for _, r in he_cells.iterrows():
        sid = str(r["sample_id"])
        he_cell_id = str(r["he_cell_id"])
        cx = float(r["x"])
        cy = float(r["y"])

        for level, patch_size in [
            (1, cfg.patch_size_px_l1),
            (2, cfg.patch_size_px_l2),
            (3, cfg.patch_size_px_l3),
        ]:
            patch_id = f"{sid}__{he_cell_id}__L{level}"
            rows.append({
                "sample_id": sid,
                "he_cell_id": he_cell_id,
                "level": level,
                f"he_patch_id_l{level}": patch_id,
                "center_x": cx,
                "center_y": cy,
                "patch_size_px": int(patch_size),
            })

    df = pd.DataFrame(rows)
    df["he_patch_id"] = df.apply(lambda x: x[f"he_patch_id_l{int(x['level'])}"], axis=1)
    return df


# ============================================================
# 3) Augmentations (DINO views)
# ============================================================

class SimpleDINOAugment(nn.Module):
    """
    Lightweight augmentation implemented with torch ops only, to minimize dependencies.

    You can later replace this with a more standard DINO-style pipeline:
    - torchvision.transforms (ColorJitter, RandomGrayscale, GaussianBlur, Solarization, etc.)
    - kornia augmentations on GPU

    These augmentations are intentionally modest to avoid destroying pathology texture:
    - random horizontal flip
    - random vertical flip
    - simple brightness/contrast jitter
    - simple Gaussian noise

    """
    def __init__(self, p_flip: float = 0.5):
        super().__init__()
        self.p_flip = p_flip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] in [0,1]
        if torch.rand(1).item() < self.p_flip:
            x = torch.flip(x, dims=[3])  # horizontal
        if torch.rand(1).item() < self.p_flip:
            x = torch.flip(x, dims=[2])  # vertical

        # brightness/contrast (simple)
        if torch.rand(1).item() < 0.8:
            b = (torch.rand(1).item() - 0.5) * 0.2  # [-0.1,0.1]
            c = 1.0 + (torch.rand(1).item() - 0.5) * 0.2  # [0.9,1.1]
            x = torch.clamp(x * c + b, 0.0, 1.0)

        # gaussian noise (simple)
        if torch.rand(1).item() < 0.2:
            noise = torch.randn_like(x) * 0.02
            x = torch.clamp(x + noise, 0.0, 1.0)
        return x


# ============================================================
# 4) Dataset: train L1 only, but optionally forward L2/L3
# ============================================================

class Phase1HEDINODataset(Dataset):
    """
    Phase 1 dataset: iterate over he_cells_table (cell-centered sampling).

    Key design points (matching our finalized plan):
    - Training objective uses L1 patches only (cell-scale).
    - L2/L3 patches may optionally be forwarded to representations (no loss, no gradients).
      This preserves interface compatibility with Phase 0 and downstream phases, and enables sanity checks.

    Output item (stable keys):
    {
      sample_id, he_cell_id,
      patch_l1, patch_l2(optional), patch_l3(optional),
      patch_id_l1/l2/l3
    }

    """
    def __init__(
        self,
        he_img_arr: np.ndarray,
        he_cells: pd.DataFrame,
        he_patches_index: pd.DataFrame,
        cfg: Phase1Config,
    ) -> None:
        super().__init__()
        self.he_img = he_img_arr
        self.he_cells = he_cells.reset_index(drop=True)
        self.he_patches_index = he_patches_index
        self.cfg = cfg

        validate_cols(self.he_cells, REQUIRED_HE_CELLS_COLS, "he_cells")

        # Build patch lookup: (sample_id, he_cell_id, level) -> (patch_id, center, size)
        self.patch_lookup: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        for _, r in self.he_patches_index.iterrows():
            sid = str(r["sample_id"])
            cid = str(r["he_cell_id"])
            level = int(r["level"])
            self.patch_lookup[(sid, cid, level)] = {
                "patch_id": str(r["he_patch_id"]),
                "center_x": float(r["center_x"]),
                "center_y": float(r["center_y"]),
                "patch_size_px": int(r["patch_size_px"]),
            }

    def __len__(self) -> int:
        return len(self.he_cells)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.he_cells.iloc[idx]
        sid = str(r["sample_id"])
        cid = str(r["he_cell_id"])

        def get_patch(level: int) -> Tuple[str, torch.Tensor]:
            info = self.patch_lookup[(sid, cid, level)]
            px = crop_patch(self.he_img, info["center_x"], info["center_y"], info["patch_size_px"])
            t = torch.from_numpy(px).permute(2, 0, 1).float() / 255.0  # [C,H,W], [0,1]
            return info["patch_id"], t

        pid1, p1 = get_patch(1)

        item: Dict[str, Any] = {
            "sample_id": sid,
            "he_cell_id": cid,
            "patch_id_l1": pid1,
            "patch_l1": p1,
        }

        if self.cfg.forward_l2_l3_for_sanity:
            pid2, p2 = get_patch(2)
            pid3, p3 = get_patch(3)
            item.update({
                "patch_id_l2": pid2,
                "patch_id_l3": pid3,
                "patch_l2": p2,
                "patch_l3": p3,
            })

        return item


def collate_phase1(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function producing a stable batch schema.

    Required:
    - patch_l1: [B,C,H,W]
    - patch_id_l1, sample_id, he_cell_id

    Optional (forward sanity only; does not participate in loss):
    - patch_l2, patch_l3

    """
    out: Dict[str, Any] = {}
    out["sample_id"] = [b["sample_id"] for b in batch]
    out["he_cell_id"] = [b["he_cell_id"] for b in batch]
    out["patch_id_l1"] = [b["patch_id_l1"] for b in batch]
    out["patch_l1"] = torch.stack([b["patch_l1"] for b in batch], dim=0)

    if "patch_l2" in batch[0]:
        out["patch_id_l2"] = [b["patch_id_l2"] for b in batch]
        out["patch_id_l3"] = [b["patch_id_l3"] for b in batch]
        out["patch_l2"] = torch.stack([b["patch_l2"] for b in batch], dim=0)
        out["patch_l3"] = torch.stack([b["patch_l3"] for b in batch], dim=0)

    return out


# ============================================================
# 5) Backbone + DINO head (lightweight implementation)
# ============================================================

class TinyConvBackbone(nn.Module):
    """
    Lightweight backbone placeholder.

    You can later replace this with a true ViT / DINOv2 ViT backbone.
    The only contract required by this script is:
    - forward(x) -> [B, embed_dim]

    Why keep a tiny conv backbone here first:
    - Phase 1's first goal is closed-loop validation: the model runs, the loss is meaningful, and it does not collapse.
    - It is faster and more robust for early testing on limited resources.
    - You can swap in a ViT later without changing the dataset/batch schema or the training loop contract.

    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DINOHead(nn.Module):
    """
    DINO projection head:
    - backbone embedding -> hidden -> output logits (out_dim).
    - The output logits are used for softmax distributions in the DINO loss.

    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ============================================================
# 6) DINO loss (student-teacher, centering, temperature schedule)
# ============================================================

class DINOLoss(nn.Module):
    """
    DINO loss (simplified but includes the essential stabilizers):
    - teacher centering (EMA of batch means)
    - teacher temperature warmup
    - fixed student temperature
    - cross-entropy between teacher probabilities and student log-probabilities

    Inputs:
    - student_logits: [B, out_dim]
    - teacher_logits: [B, out_dim] (teacher forward is detached; no gradients)

    """
    def __init__(
        self,
        out_dim: int,
        student_temp: float,
        teacher_temp: float,
        teacher_temp_warmup_steps: int,
        teacher_temp_warmup_start: float,
        center_momentum: float,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.teacher_temp_warmup_steps = teacher_temp_warmup_steps
        self.teacher_temp_warmup_start = teacher_temp_warmup_start
        self.center_momentum = center_momentum

        # Center buffer for teacher centering: maintained across steps.
        self.register_buffer("center", torch.zeros(1, out_dim))

    def get_teacher_temp(self, step: int) -> float:
        if step < self.teacher_temp_warmup_steps:
            # Linear warmup from teacher_temp_warmup_start -> teacher_temp
            t0 = self.teacher_temp_warmup_start
            t1 = self.teacher_temp
            alpha = step / max(1, self.teacher_temp_warmup_steps)
            return t0 + (t1 - t0) * alpha
        return self.teacher_temp

    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor) -> None:
        """
        Update the center as an EMA of the batch mean of teacher logits.

        This reduces collapse and stabilizes teacher distributions.

        """
        batch_center = teacher_logits.mean(dim=0, keepdim=True)  # [1,out_dim]
        self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, step: int) -> torch.Tensor:
        # Student distribution
        s = student_logits / self.student_temp
        log_p_s = torch.log_softmax(s, dim=-1)

        # Teacher distribution with centering + temperature
        t_temp = self.get_teacher_temp(step)
        t = (teacher_logits - self.center) / t_temp
        p_t = torch.softmax(t, dim=-1).detach()

        # Cross-entropy: E_{teacher}[ -log p_student ]
        loss = -(p_t * log_p_s).sum(dim=-1).mean()

        # Update centering buffer
        with torch.no_grad():
            self.update_center(teacher_logits)
        return loss


# ============================================================
# 7) LightningModule: Phase 1 training closed-loop
# ============================================================

class Phase1DINOLightningModule(L.LightningModule):
    """
    Phase 1 model:
    - student backbone + head
    - teacher backbone + head (EMA updated)
    - compute DINO loss only on L1 two views
    - optionally forward L2/L3 to produce representation statistics (no gradients)

    Outputs:
    - checkpoints: contain student network weights (teacher may be saved implicitly in the checkpoint)
    - embedding stats: norm/variance/entropy to help detect collapse

    """
    def __init__(self, cfg: Phase1Config):
        super().__init__()
        self.cfg = cfg

        # Student networks
        self.student_backbone = TinyConvBackbone(cfg.embed_dim)
        self.student_head = DINOHead(cfg.embed_dim, cfg.proj_dim, cfg.out_dim)

        # Teacher networks (same architecture; updated via EMA)
        self.teacher_backbone = TinyConvBackbone(cfg.embed_dim)
        self.teacher_head = DINOHead(cfg.embed_dim, cfg.proj_dim, cfg.out_dim)

        # Initialize teacher = student (critical for EMA start)
        self._init_teacher_from_student()

        # DINO augmentations (two views)
        self.aug = SimpleDINOAugment()

        # DINO loss module
        self.dino_loss_fn = DINOLoss(
            out_dim=cfg.out_dim,
            student_temp=cfg.student_temp,
            teacher_temp=cfg.teacher_temp,
            teacher_temp_warmup_steps=cfg.teacher_temp_warmup_steps,
            teacher_temp_warmup_start=cfg.teacher_temp_warmup_start,
            center_momentum=cfg.center_momentum,
        )

        # Explicit integer step for temperature/momentum schedules (kept separate for clarity).
        self._global_step_int = 0

    @torch.no_grad()
    def _init_teacher_from_student(self) -> None:
        """
        Initialize teacher parameters from student parameters.

        This ensures the teacher starts from the same function as the student, which is essential
        for stable early training when EMA begins.

        """
        for ps, pt in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            pt.data.copy_(ps.data)
            pt.requires_grad = False
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data.copy_(ps.data)
            pt.requires_grad = False

    def configure_optimizers(self):
        # Only optimize student parameters.
        params = list(self.student_backbone.parameters()) + list(self.student_head.parameters())
        opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return opt

    def _cosine_schedule(self, base: float, final: float, step: int, max_steps: int) -> float:
        """
        Generic cosine schedule used for teacher EMA momentum.

        As step increases, the momentum approaches 1.0, meaning the teacher becomes more stable.

        """
        if max_steps <= 1:
            return final
        t = step / (max_steps - 1)
        return final - (final - base) * (0.5 * (1.0 + math.cos(math.pi * t)))

    @torch.no_grad()
    def _update_teacher_ema(self) -> None:
        """
        EMA update:
            teacher <- m * teacher + (1-m) * student

        Momentum m is cosine scheduled from cfg.teacher_momentum to cfg.teacher_momentum_final.

        """
        m = self._cosine_schedule(
            base=self.cfg.teacher_momentum,
            final=self.cfg.teacher_momentum_final,
            step=self._global_step_int,
            max_steps=self.cfg.max_steps,
        )

        for ps, pt in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))

    @torch.no_grad()
    def _embedding_stats(self, z: torch.Tensor) -> Dict[str, float]:
        """
        Simple embedding statistics for collapse detection:
        - norm_mean: mean L2 norm across samples
        - var_mean: mean variance across embedding dimensions (near 0 is a collapse warning)
        - entropy_mean: mean softmax entropy treating z as logits (a rough but useful sanity signal)

        """
        z = z.detach()
        norm = torch.norm(z, dim=-1).mean().item()

        # Variance across batch, then mean over dimensions
        var = z.var(dim=0, unbiased=False).mean().item()

        # Softmax entropy (interpreting z as logits)
        p = torch.softmax(z, dim=-1)
        ent = (-(p * torch.log(p.clamp_min(1e-8))).sum(dim=-1)).mean().item()
        return {"norm_mean": norm, "var_mean": var, "entropy_mean": ent}

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Phase 1 training step:
        - Generate two augmented views of the same L1 patches (self-supervised).
        - Student forward on view1.
        - Teacher forward (no-grad) on view2.
        - Compute DINO loss on L1 only.
        - Optionally forward L2/L3 and log representation statistics (no loss, no gradients).
        - Save small embedding snapshots for offline sanity checks (optional).

        """
        x = batch["patch_l1"]  # [B,C,H,W]
        x = x.to(self.device)

        # Two augmented views for self-supervised training
        v1 = self.aug(x)
        v2 = self.aug(x)

        # Student forward
        s_emb = self.student_backbone(v1)            # [B, embed_dim]
        s_logits = self.student_head(s_emb)          # [B, out_dim]

        # Teacher forward (no grad)
        with torch.no_grad():
            t_emb = self.teacher_backbone(v2)
            t_logits = self.teacher_head(t_emb)

        loss_dino_l1 = self.dino_loss_fn(s_logits, t_logits, step=self._global_step_int)

        # Log the main loss
        self.log("loss_dino_l1", loss_dino_l1.detach(), prog_bar=True)

        # Optional: embedding statistics for collapse detection
        if self.cfg.log_embedding_stats_every_n_steps > 0 and (self._global_step_int % self.cfg.log_embedding_stats_every_n_steps == 0):
            stats_s = self._embedding_stats(s_emb)
            self.log("l1_student_norm_mean", stats_s["norm_mean"])
            self.log("l1_student_var_mean", stats_s["var_mean"])
            self.log("l1_student_entropy_mean", stats_s["entropy_mean"])

            # Teacher statistics are often smoother and more stable
            stats_t = self._embedding_stats(t_emb)
            self.log("l1_teacher_norm_mean", stats_t["norm_mean"])
            self.log("l1_teacher_var_mean", stats_t["var_mean"])
            self.log("l1_teacher_entropy_mean", stats_t["entropy_mean"])

        # Optional: forward L2/L3 for sanity (NO LOSS, NO GRAD)
        # This matches the final design you confirmed:
        # - L2/L3 are encoded into representations in Phase 1,
        # - but do not contribute to loss,
        # - and are primarily used later in cross-modal contrastive alignment phases.
        if self.cfg.forward_l2_l3_for_sanity and ("patch_l2" in batch) and (self._global_step_int % self.cfg.log_embedding_stats_every_n_steps == 0):
            with torch.no_grad():
                x2 = batch["patch_l2"].to(self.device)
                x3 = batch["patch_l3"].to(self.device)
                z2 = self.student_backbone(x2)
                z3 = self.student_backbone(x3)
                st2 = self._embedding_stats(z2)
                st3 = self._embedding_stats(z3)
                self.log("l2_forward_var_mean", st2["var_mean"])
                self.log("l3_forward_var_mean", st3["var_mean"])

        # Save embeddings snapshot for sanity checks (optional)
        if self.cfg.save_embeddings_sanity and self.cfg.save_embeddings_every_n_steps > 0:
            if self._global_step_int % self.cfg.save_embeddings_every_n_steps == 0:
                self._save_embedding_snapshot(batch, s_emb.detach())

        # Teacher EMA update is performed in the Lightning hook `on_after_optimizer_step`
        # to ensure it happens after the student parameters have been updated.
        return loss_dino_l1

    def on_after_optimizer_step(self, optimizer) -> None:
        """
        Lightning hook: called after the optimizer step.

        We update teacher EMA here so the teacher tracks the *updated* student.
        This avoids subtle timing issues and matches standard EMA practice.

        """
        with torch.no_grad():
            self._update_teacher_ema()
        self._global_step_int += 1

        if self.cfg.print_every_n_steps > 0 and (self._global_step_int % self.cfg.print_every_n_steps == 0):
            print(f"[Phase1] step={self._global_step_int}  (teacher EMA updated)")

    @torch.no_grad()
    def _save_embedding_snapshot(self, batch: Dict[str, Any], emb: torch.Tensor) -> None:
        """
        Save a small snapshot of embeddings plus identifiers for offline sanity checks (PCA/UMAP/collapse).

        This produces files under:
            out_dir/embeddings_sanity/l1_emb_stepXXXXXX.npz

        Stored fields:
        - sample_id, he_cell_id, patch_id_l1
        - emb: [N, embed_dim] float32

        """
        out_dir = Path(self.cfg.out_dir) / "embeddings_sanity"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save only the first N rows to keep files small.
        N = min(emb.shape[0], 256)
        data = {
            "sample_id": batch["sample_id"][:N],
            "he_cell_id": batch["he_cell_id"][:N],
            "patch_id_l1": batch["patch_id_l1"][:N],
            "emb": emb[:N].cpu().numpy().astype(np.float32),
        }
        step = self._global_step_int
        np.savez_compressed(out_dir / f"l1_emb_step{step:06d}.npz", **data)


# ============================================================
# 8) DataModule: wire data components together
# ============================================================

class Phase1DataModule(L.LightningDataModule):
    def __init__(self, paths: Phase1InputPaths, cfg: Phase1Config):
        super().__init__()
        self.paths = paths
        self.cfg = cfg

        self.dataset: Optional[Phase1HEDINODataset] = None

    def setup(self, stage: Optional[str] = None):
        set_seed(self.cfg.seed)

        # Load H&E image (demo path)
        he_img = load_image(self.paths.he_image, force_rgb=self.cfg.he_force_rgb)

        # Load he_cells_table (Phase 0 contract)
        he_cells = read_table(self.paths.he_cells_table)
        validate_cols(he_cells, REQUIRED_HE_CELLS_COLS, "he_cells")

        # Optional strict sample filtering for a single-sample demo.
        # If you later want multi-sample training, remove or generalize this filter.
        sid = self.paths.sample_id
        he_cells = he_cells[he_cells["sample_id"].astype(str) == str(sid)].copy()
        if len(he_cells) == 0:
            raise ValueError(f"No he_cells rows found for sample_id={sid}. Check your he_cells_table.")

        # Load or build he_patches_index (Phase 0 contract)
        if self.paths.he_patches_index is not None:
            he_patches_index = read_table(self.paths.he_patches_index)
        else:
            he_patches_index = build_he_patches_index_from_cells(he_cells, self.cfg)

        # Minimal schema sanity checks for he_patches_index
        needed_cols = ["sample_id", "he_cell_id", "level", "center_x", "center_y", "patch_size_px", "he_patch_id"]
        for c in needed_cols:
            if c not in he_patches_index.columns:
                raise ValueError(f"he_patches_index missing required col: {c}")

        self.dataset = Phase1HEDINODataset(
            he_img_arr=he_img,
            he_cells=he_cells,
            he_patches_index=he_patches_index,
            cfg=self.cfg,
        )

    def train_dataloader(self):
        assert self.dataset is not None
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_phase1,
            drop_last=True,
        )


# ============================================================
# 9) Main: launch Phase 1 training
# ============================================================

def main():
    # ----------------
    # Instantiate soft-coded inputs
    # ----------------
    paths = Phase1InputPaths(
        sample_id="HTA12_269",
        he_image=Path("/path/to/hta12_269_he.png"),
        he_cells_table=Path("/path/to/he_cells.parquet"),
        he_patches_index=None,  # if you saved Phase0 he_patches_index, set it here
    )

    cfg = Phase1Config(
        batch_size=64,
        max_steps=2000,
        num_workers=4,
        patch_size_px_l1=64,
        patch_size_px_l2=256,
        patch_size_px_l3=512,
        forward_l2_l3_for_sanity=True,
        out_dir=Path("./phase1_outputs"),
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    dm = Phase1DataModule(paths=paths, cfg=cfg)
    dm.setup()

    model = Phase1DINOLightningModule(cfg=cfg)

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=cfg.max_steps,
        log_every_n_steps=10,
        precision="16-mixed" if (cfg.use_amp and torch.cuda.is_available()) else 32,
        accumulate_grad_batches=cfg.grad_accum_steps,
        enable_checkpointing=True,
        default_root_dir=str(cfg.out_dir),
    )

    trainer.fit(model, datamodule=dm)

    print(f"[Phase1] Done. Checkpoints/logs saved under: {cfg.out_dir}")


if __name__ == "__main__":
    main()


# ============================================================
# 10) DOWNSTREAM USAGE NOTES (Phase 1 -> downstream phases)
# ============================================================

"""
Phase 1 outputs (artifacts) and downstream usage
------------------------------------------------

Phase 1 primarily outputs three categories of artifacts:

(1) H&E backbone checkpoint (most important)
    - Lightning will save ckpt files under default_root_dir.
    - Downstream Phase 4/5/6 should load the student_backbone weights (and optionally student_head).
    - Recommended downstream strategy:
      - Use the backbone as the feature extractor.
      - Drop the DINO head, or keep it only if you continue self-supervised regularization.

(2) L1 embedding sanity snapshots (optional)
    - out_dir/embeddings_sanity/l1_emb_stepXXXXXX.npz
    - Contains: sample_id, he_cell_id, patch_id_l1, emb
    - Intended usage:
      - PCA/UMAP to check distribution shape and detect collapse
      - Compare offline statistics to training logs (norm/variance)

(3) Training logs (loss and representation statistics)
    - Key signals to monitor:
      - loss_dino_l1 should decrease and not explode/oscillate uncontrollably
      - l1_student_var_mean should not approach ~0 (near 0 indicates collapse risk)
      - teacher stats are usually smoother; use them as a reference

How Phase 1 feeds into your 8-phase plan (alignment with your finalized diagram):

A) Phase 2 (Gene branch) / Phase 3 (Protein branch)
    - Phase 1 checkpoint does not depend on those modalities.
    - Those phases also do not require Phase 1.
    - Phase 4/5 later needs all modalities to map into a shared latent space.

B) Phase 4 (Shared latent space)
    - Add an Adapter/Projection_to_shared: H&E backbone -> shared_dim.
    - Load Phase 1 backbone checkpoint as initialization.

C) Phase 5 (first time enabling InfoNCE)
    - Level 2: align P2 (RA-scale) H&E representation with protein RA representation via InfoNCE.
    - Level 3: align P3 (spot-scale) H&E representation with gene spot representation via InfoNCE.
    - Even though Phase 1 does not train DINO on L2/L3, the backbone can already encode them.
      This is by design: L2/L3 semantics are anchored by cross-modal alignment later, not by DINO here.

D) Phase 6 (dual InfoNCE + scheduling)
    - After verifying one InfoNCE path is stable in Phase 5, enable both paths together.
    - Consider warmup or alternating schedules to reduce gradient conflict.

Engineering upgrade points (recommended later, not required for Phase 1 correctness):
- This script uses full-image numpy cropping (demo/sanity).
- For real WSI-scale training you should:
  - keep he_patches_index as the stable contract,
  - implement OpenSlide-based tile reading with caching,
  - avoid copying the full image into each DataLoader worker,
  - push augmentation into a more efficient pipeline (GPU or batched CPU).

Phase 1 termination point (what "done" means):
- You obtain a stable, cell-centric H&E backbone via L1 self-supervised DINO training.
- You preserve the ability to encode L2/L3 for downstream cross-modal alignment phases.

"""

