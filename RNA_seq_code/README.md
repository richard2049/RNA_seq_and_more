# scRNA-seq pipeline (Scanpy + scVI) — with aging/longevity analysis hooks

Reproducible single-cell RNA-seq workflow built with **Scanpy** and **scvi-tools** (SCVI + SOLO),
including QC, integration, clustering, automated/hybrid cell-type annotation, and
cell-type-specific differential expression. The repository is currently being updated for including optional
analysis modules to quantify **aging/reversal signatures** at single-cell resolution.

## Why this repo
- End-to-end workflow: raw counts → integrated embedding → clusters → labels → DE tables → figures
- Handles multi-sample integration and batch effects with scVI
- Memory-aware patterns for large matrices (sparse, backed reads, gene-block ops)

## Quickstart
### 1) Create environment
```bash
conda env create -f environment.yml
conda activate scrna_scvi
