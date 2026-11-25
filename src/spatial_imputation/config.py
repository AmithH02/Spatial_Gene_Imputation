"""Paths and per-dataset settings.

All data lives outside git. By default it is read from ``<repo>/data`` (a symlink
or folder); set ``SGI_DATA_DIR`` to point somewhere else. See ``DATA.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SGI_DATA_DIR", REPO_ROOT / "data"))

RAW_DIR = DATA_DIR / "gene_data"                    # input h5ad files
PROCESSED_DIR = DATA_DIR / "processed_gene_data"    # filtered + normalised pairs
PREDICTIONS_DIR = DATA_DIR / "predictions"          # LOO prediction matrices
RESULTS_DIR = REPO_ROOT / "results"                 # small tables / gene lists (tracked)
FIGURES_DIR = REPO_ROOT / "figures"                 # not tracked

# single-cell reference shared by both datasets (Allen ABC-mapped ageing scRNA-seq)
SC_REFERENCE_FILE = "ageing_rna_20250114.h5ad"

# spatial coordinate columns in adata.obs
X_COL, Y_COL = "center_x", "center_y"

# cell-type label in the scRNA-seq reference used for Tangram cluster mode
TANGRAM_CLUSTER_LABEL = "subclass_name"

METHODS = ("tangram", "spage")
METHOD_LABELS = {"tangram": "Tangram", "spage": "SpaGE"}


@dataclass(frozen=True)
class CortexMask:
    """Keeps cells *outside* an ellipse (i.e. the cortical ribbon of a coronal section).

    The ellipse is centred at ``(median(x) + dx, median(y) + dy)`` with semi-axes
    ``a`` and ``b``. ``exclude_box`` optionally drops an extra corner:
    cells with ``x > exclude_box[0]`` and ``y < exclude_box[1]`` are removed.
    Values were tuned by eye per section.
    """

    dx: float
    dy: float
    a: float
    b: float
    exclude_box: tuple[float, float] | None = None


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    spatial_file: str
    description: str
    cortex: CortexMask
    already_log_normalised: bool


DATASETS: dict[str, DatasetSpec] = {
    "d1": DatasetSpec(
        key="d1",
        spatial_file="ag02reg00_rawcounts.h5ad",
        description="Ageing MERFISH section ag02reg00: 40,576 cells x 500-gene panel, raw counts, "
        "ABC-atlas labels in obs.",
        cortex=CortexMask(dx=-240, dy=-280, a=1850, b=1600, exclude_box=(2400, 7950)),
        already_log_normalised=False,
    ),
    "d2": DatasetSpec(
        key="d2",
        spatial_file="SHG101350_processed.h5ad",
        description="MERFISH section SHG101350: 115,795 cells x 300-gene panel. X is already "
        "log-normalised (not raw counts).",
        cortex=CortexMask(dx=-100, dy=450, a=3300, b=2400),
        already_log_normalised=True,
    ),
}


def get_dataset(key: str) -> DatasetSpec:
    try:
        return DATASETS[key.lower()]
    except KeyError:
        raise KeyError(f"Unknown dataset '{key}'. Options: {sorted(DATASETS)}") from None


def processed_paths(key: str) -> tuple[Path, Path]:
    """(spatial, scRNA-seq) paths of the filtered + normalised AnnData pair."""
    return (
        PROCESSED_DIR / f"adata_processed_{key}.h5ad",
        PROCESSED_DIR / f"RNA_seq_adata_processed_{key}.h5ad",
    )


def predictions_path(key: str, method: str) -> Path:
    """LOO prediction matrix (cells x genes CSV) for one dataset/method."""
    return PREDICTIONS_DIR / f"tissue_{method}_preds_{key}.csv"
