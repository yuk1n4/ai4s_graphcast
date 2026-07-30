"""xarray stacking helpers for the Google Small compatible PyTorch path."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
import xarray
import yaml


DEFAULT_CONFIG = Path("configs/inference/google_graphcast_small.yaml")
PRESERVED_DIMS = ("batch", "lat", "lon")


def read_inference_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def variable_to_stacked(
    variable: xarray.Variable,
    sizes: Mapping[str, int],
    preserved_dims: tuple[str, ...] = PRESERVED_DIMS,
) -> xarray.Variable:
    """Convert one variable to ``preserved_dims + ("channels",)``.

    This mirrors Google GraphCast's ``model_utils.variable_to_stacked``. Dims not
    listed in ``preserved_dims`` are folded into the final channel axis; missing
    preserved dims are broadcast from ``sizes``.
    """

    stack_to_channels_dims = [dim for dim in variable.dims if dim not in preserved_dims]
    if stack_to_channels_dims:
        variable = variable.stack(channels=stack_to_channels_dims)
    dims = {dim: variable.sizes.get(dim) or sizes[dim] for dim in preserved_dims}
    dims["channels"] = variable.sizes.get("channels", 1)
    return variable.set_dims(dims)


def dataset_to_stacked(
    dataset: xarray.Dataset,
    sizes: Mapping[str, int] | None = None,
    preserved_dims: tuple[str, ...] = PRESERVED_DIMS,
) -> xarray.DataArray:
    """Convert an xarray Dataset to BHWC stacked channels.

    Variable order intentionally follows ``sorted(dataset.data_vars.keys())`` to
    match the Google implementation and the checkpoint data contract.
    """

    effective_sizes = sizes or dataset.sizes
    data_vars = [
        variable_to_stacked(dataset.variables[name], effective_sizes, preserved_dims)
        for name in sorted(dataset.data_vars.keys())
    ]
    coords = {
        dim: coord
        for dim, coord in dataset.coords.items()
        if dim in preserved_dims
    }
    return xarray.DataArray(
        data=xarray.Variable.concat(data_vars, dim="channels"),
        coords=coords,
    )


def select_dataset_variables(
    dataset: xarray.Dataset,
    variable_names: Iterable[str] | None,
    *,
    role: str,
) -> xarray.Dataset:
    """Return a Dataset restricted to ``variable_names`` with clear errors."""

    if variable_names is None:
        return dataset
    names = tuple(variable_names)
    missing = [name for name in names if name not in dataset.data_vars]
    if missing:
        raise KeyError(
            f"{role} dataset is missing required variables: {', '.join(missing)}"
        )
    return dataset[list(names)]


def stacked_to_dataset(
    stacked_array: xarray.Variable | xarray.DataArray,
    template_dataset: xarray.Dataset,
    preserved_dims: tuple[str, ...] = PRESERVED_DIMS,
) -> xarray.Dataset:
    """Convert BHWC stacked channels back to a Dataset shaped like template."""

    if isinstance(stacked_array, xarray.DataArray):
        stacked_variable = stacked_array.variable
    else:
        stacked_variable = stacked_array

    unstack_from_channels_sizes = {}
    var_names = sorted(template_dataset.keys())
    for name in var_names:
        template_var = template_dataset[name]
        if not all(dim in template_var.dims for dim in preserved_dims):
            raise ValueError(
                f"stacked_to_dataset requires {preserved_dims}; "
                f"{name} has dims {template_var.dims}"
            )
        unstack_from_channels_sizes[name] = {
            dim: size
            for dim, size in template_var.sizes.items()
            if dim not in preserved_dims
        }

    channels = {
        name: int(np.prod(list(unstack_sizes.values()), dtype=np.int64))
        for name, unstack_sizes in unstack_from_channels_sizes.items()
    }
    total_expected_channels = sum(channels.values())
    found_channels = stacked_variable.sizes["channels"]
    if total_expected_channels != found_channels:
        raise ValueError(
            f"expected {total_expected_channels} channels but found {found_channels}"
        )

    data_vars = {}
    index = 0
    for name in var_names:
        template_var = template_dataset[name]
        var = stacked_variable.isel({"channels": slice(index, index + channels[name])})
        index += channels[name]
        var = var.unstack({"channels": unstack_from_channels_sizes[name]})
        var = var.transpose(*template_var.dims)
        data_vars[name] = xarray.DataArray(
            data=var,
            coords=template_var.coords,
            name=template_var.name,
        )
    return type(template_dataset)(data_vars)


def inputs_to_grid_node_features(
    inputs: xarray.Dataset,
    forcings: xarray.Dataset,
    *,
    input_variables: Iterable[str] | None = None,
    forcing_variables: Iterable[str] | None = None,
) -> torch.Tensor:
    """Stack inputs and target-time forcings to ``[batch, grid_nodes, channels]``."""

    selected_inputs = select_dataset_variables(
        inputs,
        input_variables,
        role="inputs",
    )
    selected_forcings = select_dataset_variables(
        forcings,
        forcing_variables,
        role="forcings",
    )
    stacked_inputs = dataset_to_stacked(selected_inputs)
    stacked_forcings = dataset_to_stacked(selected_forcings)
    stacked = xarray.concat([stacked_inputs, stacked_forcings], dim="channels")
    stacked = stacked.transpose("batch", "lat", "lon", "channels")
    array = np.asarray(stacked.data, dtype=np.float32)
    batch, n_lat, n_lon, channels = array.shape
    return torch.from_numpy(array.reshape(batch, n_lat * n_lon, channels))


def grid_node_outputs_to_prediction(
    grid_node_outputs: torch.Tensor,
    targets_template: xarray.Dataset,
) -> xarray.Dataset:
    """Convert ``[batch, grid_nodes, channels]`` model output to target Dataset."""

    if grid_node_outputs.ndim != 3:
        raise ValueError("expected [batch, grid_nodes, channels] tensor")
    lat_size = int(targets_template.sizes["lat"])
    lon_size = int(targets_template.sizes["lon"])
    batch, grid_nodes, channels = grid_node_outputs.shape
    expected_grid_nodes = lat_size * lon_size
    if grid_nodes != expected_grid_nodes:
        raise ValueError(f"expected {expected_grid_nodes} grid nodes, got {grid_nodes}")

    array = (
        grid_node_outputs.detach()
        .cpu()
        .to(dtype=torch.float32)
        .numpy()
        .reshape(batch, lat_size, lon_size, channels)
    )
    stacked = xarray.DataArray(
        data=array,
        dims=("batch", "lat", "lon", "channels"),
        coords={
            "batch": targets_template.coords["batch"],
            "lat": targets_template.coords["lat"],
            "lon": targets_template.coords["lon"],
        },
    )
    return stacked_to_dataset(stacked, targets_template)


def build_synthetic_google_small_datasets(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    batch_size: int = 1,
    latitudes: np.ndarray | None = None,
    longitudes: np.ndarray | None = None,
) -> tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]:
    """Build deterministic inputs, targets_template, and forcings Datasets."""

    config = read_inference_config(config_path)
    variables = config["variables"]
    task_config = config["task_config"]
    pressure_levels = np.asarray(task_config["pressure_levels"], dtype=np.int32)
    input_time_steps = int(config["stacking"]["input_time_steps"])
    target_time_steps = int(config["stacking"]["target_time_steps"])
    forcing_time_steps = int(config["stacking"]["target_time_forcing_steps"])
    input_variables = tuple(task_config["input_variables"])
    target_variables = tuple(task_config["target_variables"])
    forcing_variables = tuple(task_config["forcing_variables"])
    atmospheric_variables = set(variables["atmospheric"])
    static_variables = set(task_config.get("static_variables", variables["static"]))

    if latitudes is None:
        latitudes = np.asarray([1.0, 0.0, -1.0], dtype=np.float32)
    else:
        latitudes = np.asarray(latitudes, dtype=np.float32)
    if longitudes is None:
        longitudes = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    else:
        longitudes = np.asarray(longitudes, dtype=np.float32)

    coords_common = {
        "batch": np.arange(batch_size, dtype=np.int32),
        "lat": latitudes,
        "lon": longitudes,
    }
    input_coords = {
        **coords_common,
        "time": np.arange(input_time_steps, dtype=np.int32),
        "level": pressure_levels,
    }
    target_coords = {
        **coords_common,
        "time": np.arange(target_time_steps, dtype=np.int32),
        "level": pressure_levels,
    }
    forcing_coords = {
        **coords_common,
        "time": np.arange(forcing_time_steps, dtype=np.int32),
    }

    input_vars = {}
    for name in input_variables:
        if name in atmospheric_variables:
            input_vars[name] = xarray.DataArray(
                _filled((batch_size, input_time_steps, len(latitudes), len(longitudes), len(pressure_levels)), name),
                dims=("batch", "time", "lat", "lon", "level"),
                coords=input_coords,
                name=name,
            )
        elif name in static_variables:
            input_vars[name] = xarray.DataArray(
                _filled((len(latitudes), len(longitudes)), name),
                dims=("lat", "lon"),
                coords={"lat": latitudes, "lon": longitudes},
                name=name,
            )
        else:
            input_vars[name] = xarray.DataArray(
                _filled((batch_size, input_time_steps, len(latitudes), len(longitudes)), name),
                dims=("batch", "time", "lat", "lon"),
                coords={key: input_coords[key] for key in ("batch", "time", "lat", "lon")},
                name=name,
            )

    forcing_vars = {}
    for name in forcing_variables:
        forcing_vars[name] = xarray.DataArray(
            _filled((batch_size, forcing_time_steps, len(latitudes), len(longitudes)), name),
            dims=("batch", "time", "lat", "lon"),
            coords=forcing_coords,
            name=name,
        )

    target_vars = {}
    for name in target_variables:
        if name in atmospheric_variables:
            target_vars[name] = xarray.DataArray(
                np.zeros(
                    (batch_size, target_time_steps, len(latitudes), len(longitudes), len(pressure_levels)),
                    dtype=np.float32,
                ),
                dims=("batch", "time", "lat", "lon", "level"),
                coords=target_coords,
                name=name,
            )
        else:
            target_vars[name] = xarray.DataArray(
                np.zeros((batch_size, target_time_steps, len(latitudes), len(longitudes)), dtype=np.float32),
                dims=("batch", "time", "lat", "lon"),
                coords={key: target_coords[key] for key in ("batch", "time", "lat", "lon")},
                name=name,
            )

    return (
        xarray.Dataset(input_vars),
        xarray.Dataset(target_vars),
        xarray.Dataset(forcing_vars),
    )


def _filled(shape: tuple[int, ...], name: str) -> np.ndarray:
    # Deterministic low-amplitude values keep smoke outputs reproducible.
    base = (sum(ord(char) for char in name) % 97) / 97.0
    return np.full(shape, base, dtype=np.float32)
