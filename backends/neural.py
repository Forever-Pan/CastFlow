"""Dataset-specific DLinear preparation and strict frozen-checkpoint inference."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from functools import lru_cache
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import pandas as pd

from utils.io import DatasetView, atomic_write_json, load_dataset


_TRAINING_LOCK = RLock()


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _positive(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _device_name(device: str) -> str:
    if not isinstance(device, str) or not (
        device == "cpu"
        or device == "cuda"
        or device.startswith("cuda:")
        and device[5:].isdigit()
    ):
        raise ValueError("DLinear device must be cpu, cuda, or cuda:<index>")
    return "cuda:0" if device == "cuda" else device


def _dataset(
    dataset: DatasetView | str | Path, target_col: str | None, timestamp_col: str | None
) -> tuple[DatasetView, str]:
    if isinstance(dataset, DatasetView):
        if target_col is not None and target_col != dataset.target_col:
            raise ValueError("DLinear target column differs from DatasetView")
        if timestamp_col is not None and timestamp_col != dataset.timestamp_col:
            raise ValueError("DLinear timestamp column differs from DatasetView")
        frame = dataset.frame
        values = frame[dataset.target_col].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("DLinear input targets must be finite")
        return dataset, hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    path = Path(dataset).expanduser().resolve(strict=True)
    before = _sha256(path)
    view = load_dataset(path, target_col, timestamp_col)
    if _sha256(path) != before:
        raise ValueError("DLinear source changed during preparation preflight")
    return view, before


def preflight_dlinear_inputs(
    dataset: DatasetView | str | Path,
    *,
    fit_end_row: int,
    lookback: int,
    horizon: int,
    target_col: str | None = None,
    timestamp_col: str | None = None,
    seed: int = 0,
    device: str = "cpu",
    epochs: int = 20,
    variants: int = 3,
) -> dict[str, Any]:
    for name, value in (
        ("fit_end_row", fit_end_row),
        ("lookback", lookback),
        ("horizon", horizon),
        ("epochs", epochs),
        ("variants", variants),
    ):
        _positive(value, name)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("DLinear seed must be a nonnegative integer")
    if variants > 3:
        raise ValueError(
            "DLinear has at most three distinct decomposition-kernel variants"
        )
    selected = _device_name(device)
    if selected.startswith("cuda"):
        import torch

        if (
            not torch.cuda.is_available()
            or int(selected.split(":")[1]) >= torch.cuda.device_count()
        ):
            raise ValueError("requested DLinear CUDA device is unavailable")
    view, source = _dataset(dataset, target_col, timestamp_col)
    if fit_end_row > len(view.frame):
        raise ValueError("DLinear fit_end_row exceeds the supplied training data")
    training_end = int(math.floor(fit_end_row * 0.8))
    if training_end < lookback + horizon or fit_end_row - training_end < horizon:
        raise ValueError(
            "DLinear prefix must contain complete training windows and a chronological internal-validation horizon"
        )
    kernels = list(
        dict.fromkeys(
            min(value, lookback if lookback % 2 else lookback - 1)
            for value in (25, 7, 49)
        )
    )[:variants]
    dates = (
        pd.DatetimeIndex(pd.to_datetime(view.frame[view.timestamp_col], errors="raise"))
        if view.timestamp_col
        else None
    )
    if dates is not None and (
        dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing
    ):
        raise ValueError(
            "DLinear timestamps must be present, unique, and chronological"
        )
    return {
        "source_sha256": source,
        "source_kind": "frame" if isinstance(dataset, DatasetView) else "csv",
        "source_rows": len(view.frame),
        "fit_end_row": fit_end_row,
        "training_end_row": training_end,
        "internal_validation_start_row": training_end,
        "internal_validation_end_row": fit_end_row,
        "lookback": lookback,
        "horizon": horizon,
        "target_col": view.target_col,
        "timestamp_col": view.timestamp_col,
        "kernels": kernels,
        "seed": seed,
        "epochs": epochs,
        "fit_last_timestamp": (
            dates[fit_end_row - 1].isoformat() if dates is not None else None
        ),
        "training_last_timestamp": (
            dates[training_end - 1].isoformat() if dates is not None else None
        ),
        "device": _device_name(device),
        "fit_scope": "training_prefix",
        "validation_fraction": 0.2,
        "training_dtype": "float32",
        "source_target_dtype": str(view.frame[view.target_col].dtype),
        "training_stride": max(1, horizon // 4),
        "max_training_windows": 4096,
        "max_internal_validation_windows": 1024,
        "learning_rate": 0.001,
        "batch_size": 64,
    }


def _origins(start: int, stop: int, stride: int, maximum: int) -> np.ndarray:
    result = np.arange(start, stop, stride, dtype=int)
    if len(result) > maximum:
        result = result[np.linspace(0, len(result) - 1, maximum, dtype=int)]
    return result


def _model(config: dict[str, Any]):
    import torch

    class DLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kernel = config["kernel_size"]
            self.seasonal = torch.nn.Linear(
                config["lookback"], config["horizon"], dtype=torch.float32
            )
            self.trend = torch.nn.Linear(
                config["lookback"], config["horizon"], dtype=torch.float32
            )
            for layer in (self.seasonal, self.trend):
                torch.nn.init.constant_(layer.weight, 1.0 / config["lookback"])
                torch.nn.init.zeros_(layer.bias)

        def forward(self, values):
            pad = (self.kernel - 1) // 2
            padded = torch.nn.functional.pad(
                values.unsqueeze(1), (pad, pad), mode="replicate"
            )
            trend = torch.nn.functional.avg_pool1d(
                padded, kernel_size=self.kernel, stride=1
            ).squeeze(1)
            return self.seasonal(values - trend) + self.trend(trend)

    return DLinear()


def _write_weights(path: Path, state: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".dlinear-", suffix=".safetensors", dir=path.parent
    )
    os.close(descriptor)
    try:
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in state.items()},
            temporary,
        )
        with open(temporary, "r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def dlinear_assets(folder: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(folder).expanduser().resolve()
    metadata_path, weights_path = path / "metadata.json", path / "model.safetensors"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or metadata.get("architecture") != "DLinear"
        or metadata.get("schema_version") != 1
    ):
        raise ValueError("unsupported prepared DLinear metadata")
    config = metadata.get("config", {})
    for key in ("lookback", "horizon", "kernel_size"):
        _positive(config.get(key), "DLinear." + key)
    if config["kernel_size"] % 2 != 1 or config["kernel_size"] > config["lookback"]:
        raise ValueError(
            "DLinear decomposition kernel must be odd and fit the lookback"
        )
    scaler = metadata.get("scaler", {})
    if (
        not all(
            isinstance(scaler.get(key), (int, float))
            and not isinstance(scaler[key], bool)
            and math.isfinite(scaler[key])
            for key in ("mean", "scale")
        )
        or scaler["scale"] <= 0
    ):
        raise ValueError("DLinear scaler metadata must be finite with positive scale")
    protocol = metadata.get("protocol", {})
    if (
        not isinstance(protocol, dict)
        or not isinstance(protocol.get("source_sha256"), str)
        or not 0 < protocol.get("training_end_row", 0) < protocol.get("fit_end_row", 0)
    ):
        raise ValueError("DLinear requires its training-prefix provenance")
    weight_hash = _sha256(weights_path)
    if weight_hash != metadata.get("weights_sha256"):
        raise ValueError("DLinear checkpoint checksum differs from its metadata")
    return metadata, _digest(
        {"metadata_sha256": _sha256(metadata_path), "weights_sha256": weight_hash}
    )


@contextmanager
def _preparation_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    "another process is preparing DLinear in this model store"
                ) from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(
                    "another process is preparing DLinear in this model store"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def prepare_dlinear_experts(
    dataset: DatasetView | str | Path,
    model_store: str | Path,
    *,
    fit_end_row: int,
    lookback: int,
    horizon: int,
    target_col: str | None = None,
    timestamp_col: str | None = None,
    seed: int = 0,
    device: str = "cpu",
    epochs: int = 20,
    variants: int = 3,
) -> list[dict[str, Any]]:
    """Fit fixed DLinear variants using only the declared chronological prefix."""
    kwargs = dict(
        fit_end_row=fit_end_row,
        lookback=lookback,
        horizon=horizon,
        target_col=target_col,
        timestamp_col=timestamp_col,
        seed=seed,
        device=device,
        epochs=epochs,
        variants=variants,
    )
    preflight_dlinear_inputs(dataset, **kwargs)
    with (
        _TRAINING_LOCK,
        _preparation_lock(
            Path(model_store).expanduser().resolve() / "DLinear" / ".preparation.lock"
        ),
    ):
        return _prepare_dlinear_experts(dataset, model_store, **kwargs)


def _prepare_dlinear_experts(
    dataset: DatasetView | str | Path,
    model_store: str | Path,
    *,
    fit_end_row: int,
    lookback: int,
    horizon: int,
    target_col: str | None,
    timestamp_col: str | None,
    seed: int,
    device: str,
    epochs: int,
    variants: int,
) -> list[dict[str, Any]]:
    protocol = preflight_dlinear_inputs(
        dataset,
        fit_end_row=fit_end_row,
        lookback=lookback,
        horizon=horizon,
        target_col=target_col,
        timestamp_col=timestamp_col,
        seed=seed,
        device=device,
        epochs=epochs,
        variants=variants,
    )
    import torch
    from safetensors.torch import load_file

    selected_device = torch.device(protocol["device"])
    if selected_device.type == "cuda" and (
        not torch.cuda.is_available()
        or selected_device.index >= torch.cuda.device_count()
    ):
        raise ValueError("requested DLinear CUDA device is unavailable")
    view, source = _dataset(dataset, target_col, timestamp_col)
    if source != protocol["source_sha256"]:
        raise ValueError("DLinear source changed after preparation preflight")
    values = (
        view.frame[view.target_col]
        .iloc[:fit_end_row]
        .to_numpy(dtype=np.float64, copy=True)
    )
    train_end = protocol["training_end_row"]
    mean, scale = float(np.mean(values[:train_end], dtype=np.float64)), float(
        np.std(values[:train_end], dtype=np.float64)
    )
    scale = scale if scale > 0 else 1.0
    if not math.isfinite(mean) or not math.isfinite(scale):
        raise ValueError("DLinear training-prefix scaler is nonfinite")
    normalized = ((values - mean) / scale).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("DLinear normalized training prefix exceeds float32 range")
    train_origins = _origins(
        lookback,
        train_end - horizon + 1,
        protocol["training_stride"],
        protocol["max_training_windows"],
    )
    val_origins = _origins(
        train_end,
        fit_end_row - horizon + 1,
        horizon,
        protocol["max_internal_validation_windows"],
    )
    x_train = torch.tensor(
        np.stack([normalized[i - lookback : i] for i in train_origins]),
        dtype=torch.float32,
    )
    y_train = torch.tensor(
        np.stack([normalized[i : i + horizon] for i in train_origins]),
        dtype=torch.float32,
    )
    x_val = torch.tensor(
        np.stack([normalized[i - lookback : i] for i in val_origins]),
        dtype=torch.float32,
        device=selected_device,
    )
    y_val = torch.tensor(
        np.stack([normalized[i : i + horizon] for i in val_origins]),
        dtype=torch.float32,
        device=selected_device,
    )
    prepared = []
    implementation = _sha256(Path(__file__))
    for kernel in protocol["kernels"]:
        config = {"lookback": lookback, "horizon": horizon, "kernel_size": kernel}
        identity = _digest(
            {
                "protocol": protocol,
                "config": config,
                "implementation": implementation,
                "torch": version("torch"),
                "safetensors": version("safetensors"),
            }
        )
        folder = Path(model_store).expanduser().resolve() / "DLinear" / identity[:24]
        metadata_path = folder / "metadata.json"
        if metadata_path.exists():
            metadata, asset_identity = dlinear_assets(folder)
            if (
                metadata.get("preparation_identity") != identity
                or metadata.get("protocol") != protocol
                or metadata.get("config") != config
                or metadata.get("scaler") != {"mean": mean, "scale": scale}
            ):
                raise ValueError(
                    "prepared DLinear identity conflicts with the requested training prefix"
                )
        else:
            with _TRAINING_LOCK:
                previous_threads = torch.get_num_threads()
                try:
                    if selected_device.type == "cpu":
                        torch.set_num_threads(min(4, previous_threads))
                    devices = (
                        [selected_device.index]
                        if selected_device.type == "cuda"
                        else []
                    )
                    with torch.random.fork_rng(devices=devices):
                        torch.random.default_generator.manual_seed(seed)
                        if selected_device.type == "cuda":
                            torch.cuda.default_generators[
                                selected_device.index
                            ].manual_seed(seed)
                        model = _model(config).to(selected_device)
                        optimizer = torch.optim.Adam(
                            model.parameters(), lr=protocol["learning_rate"]
                        )
                        generator = torch.Generator().manual_seed(seed)
                        best_loss, best_epoch, best_state = float("inf"), 0, None
                        for epoch in range(epochs):
                            model.train()
                            for indices in torch.randperm(
                                len(x_train), generator=generator
                            ).split(protocol["batch_size"]):
                                optimizer.zero_grad(set_to_none=True)
                                prediction = model(x_train[indices].to(selected_device))
                                loss = torch.nn.functional.mse_loss(
                                    prediction, y_train[indices].to(selected_device)
                                )
                                if not torch.isfinite(loss):
                                    raise ValueError(
                                        "DLinear training loss became nonfinite"
                                    )
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), 1.0, error_if_nonfinite=True
                                )
                                optimizer.step()
                            model.eval()
                            with torch.no_grad():
                                validation_loss = float(
                                    torch.nn.functional.mse_loss(
                                        model(x_val), y_val
                                    ).cpu()
                                )
                            if not math.isfinite(validation_loss):
                                raise ValueError(
                                    "DLinear internal-validation loss became nonfinite"
                                )
                            if validation_loss < best_loss:
                                best_loss, best_epoch = validation_loss, epoch + 1
                                best_state = {
                                    key: value.detach().cpu().clone()
                                    for key, value in model.state_dict().items()
                                }
                        if best_state is None:
                            raise ValueError(
                                "DLinear did not produce a finite checkpoint"
                            )
                        _write_weights(folder / "model.safetensors", best_state)
                finally:
                    torch.set_num_threads(previous_threads)
            metadata = {
                "schema_version": 1,
                "architecture": "DLinear",
                "config": config,
                "scaler": {"mean": mean, "scale": scale},
                "protocol": protocol,
                "preparation_identity": identity,
                "implementation_sha256": implementation,
                "weights_sha256": _sha256(folder / "model.safetensors"),
                "training_windows": len(train_origins),
                "internal_validation_windows": len(val_origins),
                "best_epoch": best_epoch,
                "internal_validation_mse": best_loss * scale * scale,
            }
            atomic_write_json(metadata_path, metadata)
            metadata, asset_identity = dlinear_assets(folder)
        # Strictly load once during preparation so an incompatible state never enters a library.
        model = _model(config)
        state = load_file(str(folder / "model.safetensors"), device="cpu")
        if any(value.dtype != torch.float32 for value in state.values()):
            raise ValueError("prepared DLinear weights must use float32")
        model.load_state_dict(state, strict=True)
        if not all(torch.isfinite(value).all() for value in model.parameters()):
            raise ValueError(
                "prepared DLinear checkpoint contains nonfinite parameters"
            )
        prepared.append(
            {
                "name": f"DLinear-k{kernel}",
                "family": "DLinear",
                "kind": "dlinear",
                "params": {},
                "is_family_default": kernel == protocol["kernels"][0],
                "device": protocol["device"],
                "model_path": str(folder),
                "model_identity": asset_identity,
                "fit_end_row": fit_end_row,
                "source_sha256": source,
                "protocol": metadata["protocol"],
            }
        )
    if _dataset(dataset, target_col, timestamp_col)[1] != source:
        raise ValueError("DLinear source changed during preparation")
    return prepared


@lru_cache(maxsize=16)
def _load_dlinear(path: str, identity: str, device: str):
    import torch
    from safetensors.torch import load_file

    metadata, actual = dlinear_assets(path)
    if not identity or actual != identity:
        raise ValueError("DLinear model differs from the frozen expert identity")
    model = _model(metadata["config"])
    state = load_file(str(Path(path) / "model.safetensors"), device="cpu")
    if any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError("prepared DLinear weights must use float32")
    model.load_state_dict(state, strict=True)
    if not all(
        np.isfinite(value.detach().cpu().numpy()).all() for value in model.parameters()
    ):
        raise ValueError("DLinear checkpoint contains nonfinite parameters")
    return model.to(device).eval().requires_grad_(False), metadata


def dlinear_forecast(
    spec: dict[str, Any], history: np.ndarray, horizon: int, *, timestamps: Any = None
) -> np.ndarray:
    import torch

    model, metadata = _load_dlinear(
        str(Path(spec["model_path"]).expanduser().resolve()),
        spec.get("model_identity", ""),
        _device_name(spec.get("device", "cpu")),
    )
    config, scaler = metadata["config"], metadata["scaler"]
    if len(history) != config["lookback"] or horizon != config["horizon"]:
        raise ValueError(
            "DLinear checkpoint dimensions differ from the current forecast request"
        )
    cutoff = metadata["protocol"].get("fit_last_timestamp")
    if cutoff is not None:
        if timestamps is None:
            raise ValueError(
                "this DLinear checkpoint requires aligned history timestamps to verify its fit cutoff"
            )
        dates = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="raise"))
        if (
            len(dates) != len(history)
            or dates.hasnans
            or dates.has_duplicates
            or not dates.is_monotonic_increasing
        ):
            raise ValueError(
                "DLinear history timestamps must be complete and chronological"
            )
        try:
            admissible = dates[-1] >= pd.Timestamp(cutoff)
        except TypeError as exc:
            raise ValueError(
                "DLinear history and training cutoff must use compatible time zones"
            ) from exc
        if not admissible:
            raise ValueError(
                "DLinear forecast origin precedes the prepared model's fit cutoff"
            )
    values = torch.as_tensor(
        (history - scaler["mean"]) / scaler["scale"],
        dtype=torch.float32,
        device=next(model.parameters()).device,
    ).unsqueeze(0)
    with torch.inference_mode():
        prediction = model(values).squeeze(0).detach().cpu().numpy().astype(float)
    result = prediction * scaler["scale"] + scaler["mean"]
    if result.shape != (horizon,) or not np.isfinite(result).all():
        raise ValueError("DLinear produced an invalid forecast")
    return result
