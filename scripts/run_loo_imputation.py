"""Run leave-one-gene-out imputation headless (slow: hours for Tangram on all genes).

    python scripts/run_loo_imputation.py --dataset d1 --method spage
    python scripts/run_loo_imputation.py --dataset d2 --method tangram --genes nos1 ngb

If the processed pair for the dataset does not exist yet it is built from the raw
files first (cortex filter + normalisation) and saved to data/processed_gene_data/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_imputation import config, data  # noqa: E402
from spatial_imputation.imputation import loo_impute  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(config.DATASETS))
    ap.add_argument("--method", required=True, choices=config.METHODS)
    ap.add_argument("--genes", nargs="*", help="subset of genes (default: all shared genes)")
    ap.add_argument("--out", type=Path, help="output CSV (default: data/predictions/tissue_<method>_preds_<dataset>.csv)")
    args = ap.parse_args()

    sp_path, _ = config.processed_paths(args.dataset)
    if sp_path.exists():
        spatial, reference = data.load_processed(args.dataset)
    else:
        print(f"Building processed data for {args.dataset} from raw files")
        spatial, reference = data.load_raw(args.dataset)
        spatial = data.filter_cortex(spatial, args.dataset)
        spatial, reference = data.preprocess_pair(spatial, reference)
        data.save_processed(args.dataset, spatial, reference)

    genes = [g.lower() for g in args.genes] if args.genes else None
    preds = loo_impute(spatial, reference, args.method, genes=genes)

    out = args.out or config.predictions_path(args.dataset, args.method)
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out)
    print(f"Wrote {preds.shape[0]} cells x {preds.shape[1]} genes to {out}")


if __name__ == "__main__":
    main()
