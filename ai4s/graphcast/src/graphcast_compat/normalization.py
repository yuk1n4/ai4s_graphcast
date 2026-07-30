"""Google-style normalization helpers for the compatible PyTorch path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import xarray as xr


STDDEV_FILE = "stddev_by_level.nc"
MEAN_FILE = "mean_by_level.nc"
DIFFS_STDDEV_FILE = "diffs_stddev_by_level.nc"
REQUIRED_STATS_FILES = (STDDEV_FILE, MEAN_FILE, DIFFS_STDDEV_FILE)


@dataclass(frozen=True)
class NormalizationStats:
    stddev_by_level: xr.Dataset
    mean_by_level: xr.Dataset
    diffs_stddev_by_level: xr.Dataset
    source_dir: Path


def load_normalization_stats(stats_dir: str | Path) -> NormalizationStats:
    """Load Google GraphCast normalization stats from a directory."""

    path = Path(stats_dir)
    missing = [name for name in REQUIRED_STATS_FILES if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing normalization stats in {path}: {', '.join(missing)}"
        )
    return NormalizationStats(
        stddev_by_level=xr.open_dataset(path / STDDEV_FILE).load(),
        mean_by_level=xr.open_dataset(path / MEAN_FILE).load(),
        diffs_stddev_by_level=xr.open_dataset(path / DIFFS_STDDEV_FILE).load(),
        source_dir=path,
    )


def stats_file_status(stats_dir: str | Path) -> dict[str, bool]:
    path = Path(stats_dir)
    return {name: (path / name).exists() for name in REQUIRED_STATS_FILES}


def missing_stats_files(stats_dir: str | Path) -> list[str]:
    status = stats_file_status(stats_dir)
    return [name for name, exists in status.items() if not exists]


def normalize_dataset(
    values: xr.Dataset,
    scales: xr.Dataset,
    locations: xr.Dataset | None,
) -> xr.Dataset:
    """Normalize a Dataset by variable name, matching Google semantics."""

    data_vars = {}
    for name, array in values.data_vars.items():
        result = array
        if locations is not None and name in locations:
            result = result - locations[name].astype(result.dtype)
        if name in scales:
            result = result / scales[name].astype(result.dtype)
        data_vars[name] = result
    return xr.Dataset(data_vars)


def unnormalize_dataset(
    values: xr.Dataset,
    scales: xr.Dataset,
    locations: xr.Dataset | None,
) -> xr.Dataset:
    """Invert ``normalize_dataset`` by variable name."""

    data_vars = {}
    for name, array in values.data_vars.items():
        result = array
        if name in scales:
            result = result * scales[name].astype(result.dtype)
        if locations is not None and name in locations:
            result = result + locations[name].astype(result.dtype)
        data_vars[name] = result
    return xr.Dataset(data_vars)


def unnormalize_prediction_and_add_input(
    inputs: xr.Dataset,
    norm_prediction: xr.Dataset,
    stats: NormalizationStats,
) -> xr.Dataset:
    """Invert Google ``InputsAndResiduals`` prediction transform."""

    data_vars = {}
    for name, prediction in norm_prediction.data_vars.items():
        single_prediction = xr.Dataset({name: prediction})
        if name in inputs:
            unnormalized = unnormalize_dataset(
                single_prediction,
                stats.diffs_stddev_by_level,
                locations=None,
            )[name]
            last_input = inputs[name].isel(time=-1)
            data_vars[name] = unnormalized + last_input
        else:
            data_vars[name] = unnormalize_dataset(
                single_prediction,
                stats.stddev_by_level,
                stats.mean_by_level,
            )[name]
    return xr.Dataset(data_vars)


def predict_with_inputs_and_residuals(
    model,
    inputs: xr.Dataset,
    targets_template: xr.Dataset,
    forcings: xr.Dataset,
    stats: NormalizationStats,
) -> xr.Dataset:
    """Run model prediction with Google ``InputsAndResiduals`` normalization."""

    norm_inputs = normalize_dataset(
        inputs,
        scales=stats.stddev_by_level,
        locations=stats.mean_by_level,
    )
    norm_forcings = normalize_dataset(
        forcings,
        scales=stats.stddev_by_level,
        locations=stats.mean_by_level,
    )
    norm_prediction = model.predict_xarray(norm_inputs, targets_template, norm_forcings)
    return unnormalize_prediction_and_add_input(inputs, norm_prediction, stats)


def missing_variables_for_dataset(
    dataset: xr.Dataset,
    stats_dataset: xr.Dataset,
) -> list[str]:
    return sorted(name for name in dataset.data_vars if name not in stats_dataset)


def describe_stats_coverage(
    *,
    inputs: xr.Dataset,
    targets_template: xr.Dataset,
    forcings: xr.Dataset,
    stats: NormalizationStats,
) -> dict[str, list[str]]:
    """Return missing variable coverage for Google normalization stats."""

    merged_inputs = xr.merge([inputs, forcings], compat="override")
    return {
        "stddev_missing_for_inputs_or_forcings": missing_variables_for_dataset(
            merged_inputs,
            stats.stddev_by_level,
        ),
        "mean_missing_for_inputs_or_forcings": missing_variables_for_dataset(
            merged_inputs,
            stats.mean_by_level,
        ),
        "diffs_stddev_missing_for_targets": missing_variables_for_dataset(
            targets_template,
            stats.diffs_stddev_by_level,
        ),
    }


def coverage_ok(coverage: dict[str, Iterable[str]]) -> bool:
    return all(not list(values) for values in coverage.values())
