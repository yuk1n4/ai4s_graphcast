"""PyTorch compatibility modules for original GraphCast checkpoints."""

from .data import (
    build_synthetic_google_small_datasets,
    dataset_to_stacked,
    grid_node_outputs_to_prediction,
    inputs_to_grid_node_features,
    select_dataset_variables,
    stacked_to_dataset,
)
from .graph import GoogleSmallGraph, build_google_small_graph
from .google_small import (
    GoogleSmallConfig,
    GoogleSmallCompatibleModel,
    build_google_small_compatible_model,
)
from .backend import (
    BackendProfile,
    FlagOSState,
    RuntimeBackend,
    backend_report_lines,
    memory_text,
    reset_peak_memory,
    resolve_runtime_backend,
    synchronize,
)
from .normalization import (
    DIFFS_STDDEV_FILE,
    MEAN_FILE,
    REQUIRED_STATS_FILES,
    STDDEV_FILE,
    NormalizationStats,
    describe_stats_coverage,
    load_normalization_stats,
    missing_stats_files,
    predict_with_inputs_and_residuals,
)

__all__ = [
    "GoogleSmallGraph",
    "GoogleSmallConfig",
    "GoogleSmallCompatibleModel",
    "BackendProfile",
    "FlagOSState",
    "RuntimeBackend",
    "backend_report_lines",
    "build_synthetic_google_small_datasets",
    "build_google_small_graph",
    "build_google_small_compatible_model",
    "dataset_to_stacked",
    "grid_node_outputs_to_prediction",
    "inputs_to_grid_node_features",
    "select_dataset_variables",
    "DIFFS_STDDEV_FILE",
    "MEAN_FILE",
    "NormalizationStats",
    "REQUIRED_STATS_FILES",
    "STDDEV_FILE",
    "describe_stats_coverage",
    "load_normalization_stats",
    "missing_stats_files",
    "memory_text",
    "predict_with_inputs_and_residuals",
    "reset_peak_memory",
    "resolve_runtime_backend",
    "stacked_to_dataset",
    "synchronize",
]
