# Gene imputation in MERFISH spatial transcriptomics (Tangram vs SpaGE)

LIDo PhD rotation 2 (2025). How well can genes missing from a MERFISH panel be imputed from a scRNA-seq reference, and can we trust the uncertainty estimates? Two imputation methods, **Tangram** and **SpaGE**, are benchmarked by leave-one-gene-out (LOO) cross-validation on two mouse-brain cortex sections, and **TISSUE** is used to calibrate prediction intervals.

Write-up: *Evaluating Gene Imputation Algorithms in MERFISH Spatial Transcriptomics* (not in this repo).

## Layout

```
notebooks/   01-04 analysis notebooks (thin: settings, function calls, figures, explanations)
scripts/     headless runner for the slow LOO imputation step
src/spatial_imputation/   reusable code (imported by both)
tests/       unit tests for metrics, statistics and preprocessing
results/     small outputs: per-gene metrics, significant gene lists
data/        inputs and intermediate files (not tracked, see DATA.md)
```

## Pipeline

| Notebook | What it does | Main outputs |
|---|---|---|
| `01_data_and_preprocessing` | Cortex filter, sequencing-depth comparison, log-normalisation, shared genes, sparsity | `data/processed_gene_data/*_{d1,d2}.h5ad` |
| `02_loo_imputation` | Leave-one-gene-out predictions for every shared gene (Tangram cluster mode, SpaGE) | `data/predictions/tissue_{method}_preds_{dataset}.csv` |
| `03_imputation_benchmark` | PCC, Spearman, RMSE, cosine, JS, SSIM; permutation significance (BH-FDR); Tangram vs SpaGE; D1 vs D2; Moran's I and sparsity as determinants | `results/gene_metrics.csv`, `results/significant_genes/` |
| `04_uncertainty_calibration` | TISSUE spatial graph, conformal scores, 77% prediction intervals; does PI width track error? | figures |

Run in order. 01 and 02 are slow and only need re-running if the inputs change; their outputs are cached in `data/`. 02 can also run without Jupyter:

```bash
python scripts/run_loo_imputation.py --dataset d1 --method tangram
python scripts/run_loo_imputation.py --dataset d2 --method spage
```

## Code

`src/spatial_imputation/`:

| Module | Contents |
|---|---|
| `config` | data paths (`SGI_DATA_DIR`), dataset definitions and cortex-mask parameters |
| `data` | loading, cortex filter, preprocessing, sparsity, loading predictions |
| `imputation` | Tangram / SpaGE wrappers and the LOO loop |
| `metrics` | per-gene metrics, sparsity-weighted score, score-vs-sparsity AUC |
| `stats` | vectorised permutation test, Wilcoxon / Mann-Whitney comparisons, Moran's I, set overlap |
| `uncertainty` | TISSUE calibration, error-vs-uncertainty reliability, quartile analysis |
| `plotting` | all figures |

Notebooks add `src/` to `sys.path` and `chdir` to the repo root, so they run from `notebooks/` or the root. Alternatively `pip install -e .`.

## Setup

```bash
conda env create -f environment.yml        # env: spatial_imputation (installs this package + Tangram/TISSUE)
conda activate spatial_imputation
pytest                                     # unit tests (no data needed)
```

Notebook 03 and the tests only need the core dependencies; 01, 02 and 04 also need the `impute` extras (`tangram-sc==1.0.3`, `tissue-sc`, `squidpy`, `torch`).

Data are not tracked. Put them in `data/` (or a symlink) or set `SGI_DATA_DIR`; see [DATA.md](DATA.md).

## Main results

| | Tangram D1 | SpaGE D1 | Tangram D2 | SpaGE D2 |
|---|---|---|---|---|
| cells x genes | 19,441 x 464 | 19,441 x 464 | 51,377 x 296 | 51,377 x 296 |
| mean PCC | 0.314 | 0.315 | 0.294 | 0.262 |
| mean SSIM | 0.527 | 0.648 | 0.501 | 0.414 |
| mean JS (lower = better) | 0.405 | 0.396 | 0.126 | 0.168 |
| genes significant (FDR < 0.05) | 447 | 445 | 281 | 281 |

- Neither method is better everywhere: SpaGE is ahead on most metrics in D1 (largest gap in SSIM), Tangram on all four in D2. ~96% of genes are imputed better than chance by both, with ~97% overlap.
- How well a gene can be imputed depends mostly on its spatial structure: Moran's I correlates with accuracy (Spearman) at r ≈ 0.7 for both methods in both sections. Sparsity matters strongly in D1 but not in D2.
- JS divergence follows gene sparsity almost exactly (r = 0.85-0.98), so it says more about the gene than the method.
- TISSUE prediction-interval width barely tracks real error (D2: mean per-gene ρ(|error|, PI width) = 0.06 SpaGE, 0.11 Tangram), so its uncertainty is not a reliable guide for unmeasured genes.

## Notes and known caveats

- **Double normalisation of D2.** `SHG101350_processed.h5ad` is already log-normalised, but preprocessing applies `normalize_total` + `log1p` again (as in the original analysis). Correlation-based metrics are largely unaffected; RMSE and the absolute scale are. To fix, pass `normalize=False` for the spatial object of D2 and re-run 01-04.
- The cortex mask is an ellipse hand-tuned per section (`config.DATASETS`); it is not an anatomical annotation.
- SSIM is the global (non-windowed) form on max-scaled vectors, matching the write-up.
- The permutation test is seeded (`seed=0`) and vectorised; it reproduces the saved significant-gene lists exactly.

## History

The pre-refactor notebooks (`tangram_imputation`, `spage_imputation`, `tissue_imputation*`, `imputation_analysis*`, `combined_dataset_analysis`) are kept at tag `pre-refactor`. The refactor is on branch `refactor/src-layout`.
