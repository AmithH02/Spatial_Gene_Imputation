# Data

No data are tracked in git. The code reads from `data/` in the repo root (a folder or symlink), or from the folder named by the `SGI_DATA_DIR` environment variable.

```
data/
├── gene_data/                     inputs (notebook 01)
│   ├── ag02reg00_rawcounts.h5ad       D1 spatial      163 MB
│   ├── SHG101350_processed.h5ad       D2 spatial      144 MB
│   └── ageing_rna_20250114.h5ad       scRNA-seq ref   2.4 GB
├── processed_gene_data/           written by 01, read by 02-04
│   ├── adata_processed_{d1,d2}.h5ad
│   └── RNA_seq_adata_processed_{d1,d2}.h5ad
└── predictions/                   written by 02, read by 03-04
    └── tissue_{tangram,spage}_preds_{d1,d2}.csv
```

If you have the processed files and predictions, you can run notebooks 03 and 04 without the raw inputs.

## Files used

| Key | File | Shape (cells x genes) | Contents |
|---|---|---|---|
| D1 | `ag02reg00_rawcounts.h5ad` | 40,576 x 500 | MERFISH coronal section, **raw integer counts**. `obs`: `center_x`, `center_y`, `transcript_count`, `volume`, ABC-atlas labels (`class_name`, `subclass_name`, `supertype_name`, ...) |
| D2 | `SHG101350_processed.h5ad` | 115,795 x 300 | MERFISH coronal section, **already log-normalised** (X max ≈ 9). `obs`: `center_x`, `center_y`, `n_counts`, `n_genes`, `leiden`; `var`: `mean`, `std` |
| Reference | `ageing_rna_20250114.h5ad` | 173,199 x 19,957 | Ageing mouse-brain scRNA-seq, counts (sparse CSC). `obs` includes `sample`, `celltype`, doublet scores and ABC-atlas labels (`class_name`, `subclass_name`, ...) |

After the cortex filter and gene matching: D1 = 19,441 cells x 464 genes, D2 = 51,377 cells x 296 genes.

Provenance (who generated each section, and which ABC-atlas mapping version was used for the labels) is not recorded in the files. Add it here.

## Using other data

Any spatial / single-cell pair works if it matches these requirements:

**Spatial AnnData (`.h5ad`)**
- cells x genes, gene **symbols** in `var_names` (case does not matter, everything is lower-cased)
- `obs['center_x']`, `obs['center_y']`: cell centroids in a common unit (µm). Other column names: change `X_COL` / `Y_COL` in `config.py`
- `X` should be **raw counts** (it gets `normalize_total` + `log1p`). If it is already normalised, call `data.preprocess_pair(..., normalize=False)`
- optional `obs['transcript_count']` (used only for the depth plot)
- any imaging platform with single-cell segmentation works (MERFISH, Xenium, CosMx). Spot-based data (Visium) runs, but Tangram cluster mode and the cortex filter assume single cells

**scRNA-seq reference AnnData**
- cells x genes counts with gene symbols overlapping the spatial panel (a few hundred shared genes is typical)
- `obs['subclass_name']` (or another cell-type column; change `TANGRAM_CLUSTER_LABEL` in `config.py`) for Tangram cluster mode
- ideally the same species, region and age range as the section

**To add a dataset**, add an entry to `config.DATASETS` with its file name and cortex-mask parameters (or a mask that keeps everything, e.g. `CortexMask(0, 0, a=1e-9, b=1e-9)`), then run notebooks 01-04 with the new key.

## Not needed

These files in the old working folder are earlier work and are not used by this repo:

- `old/old_files/cells_mapped.h5ad` (10.7 GB), `ad_ge.h5ad` (6.5 GB), `cells_subset_mapped.h5ad` (3.1 GB), `subclass_mapped.h5ad`, `supertype_mapped.h5ad`, `mapped.h5ad`: Tangram mapping and projection outputs from the first stand-alone Tangram notebooks (late April 2025)
- `predictions/old_tissue_*_preds_d2.csv`: D2 predictions from 28 May / 11 June 2025, before the D2 cortex filter was fixed (the current ones are from 12-13 June)
- `Imputation/spage_preds_d1.csv`, `old/old_files/spage_preds_d1_non_standardised.csv`: early SpaGE runs
- `Imputation/gene_data/ageing02reg00_processed_20241128.h5ad`: earlier processed version of the D1 section, used only in the first exploration notebook
