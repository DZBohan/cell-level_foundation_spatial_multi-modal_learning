# A Cell-Level, Multi-Scale Foundation Framework for Spatial Multi-Modal Learning

## Index

Phase 0. Data Contract & Project Skeleton

Phase 1. H&E Image Branch (Self-Supervised)

Phase 2. Gene Branch (Masked Reconstruction with Spatial Context)

Phase 3. Protein Branch

Phase 4. Shared Latent Space & Contrastive Projection Layer

Phase 5. Single-Scale Cross-Modal Contrastive Alignment

Phase 6. Multi-Objective Contrastive Stabilization

Phase 7. Registration-Aware Ground Truth Integration

# Phase 0. Data Contract & Project Skeleton

Phase 0 defines the engineering contract and runnable skeleton for the entire multi-modal spatial foundation framework.

Its sole purpose is to prove that all planned modalities, scales, and losses can be wired together in a stable, registration-agnostic way, and that the full training loop can run end-to-end without ambiguity.

1. Explicit data contracts for all modalities: H&E (image + cell coordinates)l; Visium (spot-level gene counts); CODEX (cell-level protein features + RA queries); optional pairing tables for future contrastive learning.

2. Stable batch schema: Every batch has the same keys; optional modalities are represented via masks, not missing tensors.

3. Soft-coded multi-scale patch system: Three H&E patch levels L1 (cell-scale) / L2 (RA-scale) / L3 (spot-scale).

4. Loss routing logic: InfoNCE losses are automatically enabled or disabled; loss computation depends only on data presence, not code branching.

5. Runnable dummy training loop: A minimal LightningModule runs N steps; confirms that all components are compatible.

# Phase 1 — H&E Image Branch (Self-Supervised)

Phase 1 trains the H&E image backbone using self-supervised learning, producing a general-purpose visual representation that can later be aligned with gene and protein modalities.