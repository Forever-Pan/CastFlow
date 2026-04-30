from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    train_path: str
    test_path: str
    lookback: int
    horizon: int
    seasonal_period: int
    memory_stride: int
    anchor_stride: int


DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"


DATASETS: dict[str, DatasetSpec] = {
    "BE": DatasetSpec("BE", "train/EPF_BE_train_val.csv", "test/EPF_BE_test.csv", 168, 24, 24, 48, 48),
    "DE": DatasetSpec("DE", "train/EPF_DE_train_val.csv", "test/EPF_DE_test.csv", 168, 24, 24, 48, 48),
    "FR": DatasetSpec("FR", "train/EPF_FR_train_val.csv", "test/EPF_FR_test.csv", 168, 24, 24, 48, 48),
    "NP": DatasetSpec("NP", "train/EPF_NP_train_val.csv", "test/EPF_NP_test.csv", 168, 24, 24, 48, 48),
    "PJM": DatasetSpec("PJM", "train/EPF_PJM_train_val.csv", "test/EPF_PJM_test.csv", 168, 24, 24, 48, 48),
    "ETTh1": DatasetSpec("ETTh1", "train/ETT_ETTh1_train_val.csv", "test/ETT_ETTh1_test.csv", 96, 96, 24, 48, 24),
    "ETTm1": DatasetSpec("ETTm1", "train/ETT_ETTm1_train_val.csv", "test/ETT_ETTm1_test.csv", 96, 96, 24, 96, 24),
    "MOPEX": DatasetSpec("MOPEX", "train/mopex_train_val.csv", "test/mopex_test.csv", 96, 96, 24, 48, 24),
    "SP": DatasetSpec("SP", "train/sunny_power_train_val.csv", "test/sunny_power_test.csv", 96, 96, 24, 96, 24),
    "WP": DatasetSpec("WP", "train/windy_power_train_val.csv", "test/windy_power_test.csv", 96, 96, 24, 96, 24),
}


def get_dataset(name: str) -> DatasetSpec:
    key = normalize_dataset_name(name)
    if key not in DATASETS:
        available = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown dataset {name!r}; available: {available}")
    return DATASETS[key]


def normalize_dataset_name(name: str) -> str:
    mapping = {
        "sunny": "SP",
        "sunny_power": "SP",
        "solar": "SP",
        "windy": "WP",
        "windy_power": "WP",
        "wind": "WP",
        "mopex": "MOPEX",
        "ettm1": "ETTm1",
        "etth1": "ETTh1",
    }
    return mapping.get(name, name)


def resolve_train_path(spec: DatasetSpec) -> Path:
    return DATA_ROOT / spec.train_path


def resolve_test_path(spec: DatasetSpec) -> Path:
    return DATA_ROOT / spec.test_path


def infer_dataset_from_path(path: str | Path) -> DatasetSpec | None:
    candidate = Path(path)
    text = str(candidate).lower()
    filename = candidate.name.lower()
    stems = {candidate.stem.lower(), filename}
    for spec in DATASETS.values():
        train_name = Path(spec.train_path).name.lower()
        test_name = Path(spec.test_path).name.lower()
        train_stem = Path(spec.train_path).stem.lower()
        test_stem = Path(spec.test_path).stem.lower()
        aliases = {
            spec.name.lower(),
            train_name,
            test_name,
            train_stem,
            test_stem,
        }
        if filename in aliases or stems & aliases:
            return spec
        if any(alias in text for alias in aliases):
            return spec
    return None


def default_window_stride(name: str | None, kind: str) -> int | None:
    if not name:
        return None
    spec = get_dataset(name)
    if kind == "anchorer":
        return spec.anchor_stride
    if kind == "memory":
        return spec.memory_stride
    raise ValueError(f"unknown stride kind: {kind}")
