"""Backend and FlagOS runtime helpers for GraphCast inference.

The model code should stay shared across chips. This module owns the runtime
differences: device import/checks, synchronization, memory stats, and optional
FlagOS operator layers.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.metadata as metadata
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Callable

import torch


SAFE_FLAGGEMS_OPS = ("silu", "add")
RISKY_FLAGGEMS_OPS = (
    "cat",
    "addmm",
    "index",
    "index_add_",
    "layer_norm",
    "scatter_add_",
    "scatter_add",
)


@dataclass(frozen=True)
class BackendProfile:
    name: str
    device_type: str
    default_device: str
    runtime_family: str
    chip: str
    supports_flaggems: bool = False
    supports_flagtree: bool = False
    flaggems_vendor: str | None = None
    container_image: str | None = None
    status: str = "active"
    notes: str = ""


@dataclass(frozen=True)
class FlagOSState:
    mode: str
    strict: bool
    requested_ops: tuple[str, ...]
    status: str
    vendor: str | None = None
    record_path: str | None = None
    bishengir_opt_dir: str | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None
    patched_ops: tuple[str, ...] = ()
    registered_ops: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeBackend:
    profile: BackendProfile
    device: torch.device
    flagos: FlagOSState
    versions: dict[str, str]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "backend": self.profile.name,
            "device": str(self.device),
            "device_type": self.profile.device_type,
            "runtime_family": self.profile.runtime_family,
            "chip": self.profile.chip,
            "status": self.profile.status,
            "flagos_mode": self.flagos.mode,
            "flagos_status": self.flagos.status,
            "flagos_strict": self.flagos.strict,
            "flaggems_ops": list(self.flagos.requested_ops),
            "flaggems_vendor": self.flagos.vendor,
            "flaggems_record_path": self.flagos.record_path,
            "flaggems_source_root": os.environ.get("GRAPHCAST_FLAGGEMS_SOURCE_ROOT"),
            "flaggems_module_origin": _module_origin("flag_gems"),
            "bishengir_opt_dir": self.flagos.bishengir_opt_dir,
            "gather_backend": os.environ.get("GRAPHCAST_GATHER_BACKEND", "advanced_index"),
            "flagos_warnings": list(self.flagos.warnings),
            "flagos_error": self.flagos.error,
            "flaggems_patched_ops": list(self.flagos.patched_ops),
            "flaggems_registered_ops": list(self.flagos.registered_ops),
            "versions": dict(self.versions),
        }


BACKEND_PROFILES: dict[str, BackendProfile] = {
    "cpu": BackendProfile(
        name="cpu",
        device_type="cpu",
        default_device="cpu",
        runtime_family="pytorch",
        chip="host",
        notes="CPU smoke path only; not suitable for full 25km inference.",
    ),
    "native_cuda": BackendProfile(
        name="native_cuda",
        device_type="cuda",
        default_device="cuda",
        runtime_family="pytorch_cuda",
        chip="nvidia",
        notes="NVIDIA H100 correctness/performance baseline on CUDA.",
    ),
    "native_hygon": BackendProfile(
        name="native_hygon",
        device_type="cuda",
        default_device="cuda",
        runtime_family="pytorch_hip",
        chip="hygon",
        notes=(
            "Hygon/BW PyTorch eager baseline. The model uses the torch.cuda "
            "compatibility API, while the runtime is HIP/DAS/DTK underneath."
        ),
    ),
    "native_ptg": BackendProfile(
        name="native_ptg",
        device_type="cuda",
        default_device="cuda",
        runtime_family="pytorch_ppu",
        chip="pingtouge_810e",
        notes="PTG 810E PyTorch eager baseline through the torch.cuda-compatible PPU runtime.",
    ),
    "native_mthreads": BackendProfile(
        name="native_mthreads",
        device_type="musa",
        default_device="musa",
        runtime_family="pytorch_musa",
        chip="mthreads_s5000",
        notes="Moore Threads S5000 PyTorch eager baseline through torch_musa.",
    ),
    "native_metax": BackendProfile(
        name="native_metax",
        device_type="cuda",
        default_device="cuda",
        runtime_family="pytorch_metax",
        chip="metax_c550",
        notes="MetaX C550 PyTorch eager baseline through its torch.cuda-compatible runtime.",
    ),
    "native_npu": BackendProfile(
        name="native_npu",
        device_type="npu",
        default_device="npu",
        runtime_family="pytorch_npu",
        chip="huawei_ascend",
        notes="Ascend 910C correctness baseline on torch_npu eager.",
    ),
    "flagos_hygon": BackendProfile(
        name="flagos_hygon",
        device_type="cuda",
        default_device="cuda",
        runtime_family="flagos",
        chip="hygon",
        supports_flaggems=True,
        supports_flagtree=False,
        flaggems_vendor="hygon",
        notes=(
            "FlagOS profile for Hygon/BW. FlagGems is optional and uses the "
            "torch.cuda-compatible HIP/DAS/DTK runtime underneath."
        ),
    ),
    "flagos_ptg": BackendProfile(
        name="flagos_ptg",
        device_type="cuda",
        default_device="cuda",
        runtime_family="flagos",
        chip="pingtouge_810e",
        supports_flaggems=True,
        supports_flagtree=False,
        flaggems_vendor="thead",
        notes="FlagOS profile for PTG 810E on the torch.cuda-compatible PPU runtime.",
    ),
    "flagos_mthreads": BackendProfile(
        name="flagos_mthreads",
        device_type="musa",
        default_device="musa",
        runtime_family="flagos",
        chip="mthreads_s5000",
        supports_flaggems=True,
        supports_flagtree=False,
        flaggems_vendor="mthreads",
        notes="FlagOS profile for Moore Threads S5000 through torch_musa.",
    ),
    "flagos_metax": BackendProfile(
        name="flagos_metax",
        device_type="cuda",
        default_device="cuda",
        runtime_family="flagos",
        chip="metax_c550",
        supports_flaggems=True,
        supports_flagtree=False,
        flaggems_vendor="metax",
        notes="FlagOS profile for MetaX C550 through its torch.cuda-compatible runtime.",
    ),
    "flagos_ascend": BackendProfile(
        name="flagos_ascend",
        device_type="npu",
        default_device="npu",
        runtime_family="flagos",
        chip="huawei_ascend",
        supports_flaggems=True,
        supports_flagtree=False,
        flaggems_vendor="ascend",
        notes="FlagOS profile for Ascend 910C; FlagGems is optional and off by default.",
    ),
    "flagos_nvidia": BackendProfile(
        name="flagos_nvidia",
        device_type="cuda",
        default_device="cuda",
        runtime_family="flagos",
        chip="nvidia",
        supports_flaggems=True,
        supports_flagtree=False,
        flaggems_vendor="nvidia",
        notes="FlagOS profile for NVIDIA H100; FlagGems is optional and off by default.",
    ),
}

BACKEND_ALIASES = {
    "auto": "auto",
    "cuda": "native_cuda",
    "nvidia": "native_cuda",
    "hygon": "native_hygon",
    "bw1000": "native_hygon",
    "hip": "native_hygon",
    "hygon_native": "native_hygon",
    "hygon_flagos": "flagos_hygon",
    "flagos_hygon": "flagos_hygon",
    "ptg": "native_ptg",
    "810e": "native_ptg",
    "pingtouge": "native_ptg",
    "ptg_native": "native_ptg",
    "ptg_flagos": "flagos_ptg",
    "flagos_ptg": "flagos_ptg",
    "flagos_pingtouge": "flagos_ptg",
    "musa": "native_mthreads",
    "mthreads": "native_mthreads",
    "s5000": "native_mthreads",
    "mthreads_native": "native_mthreads",
    "mthreads_flagos": "flagos_mthreads",
    "flagos_mthreads": "flagos_mthreads",
    "metax": "native_metax",
    "c550": "native_metax",
    "metax_native": "native_metax",
    "metax_flagos": "flagos_metax",
    "flagos_metax": "flagos_metax",
    "npu": "native_npu",
    "ascend_native": "native_npu",
    "ascend": "flagos_ascend",
    "flagos": "flagos_ascend",
    "nvidia_flagos": "flagos_nvidia",
    "flagos_cuda": "flagos_nvidia",
    "flagos_nvidia": "flagos_nvidia",
    "cpu": "cpu",
}


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


def _cuda_backend_profile() -> str:
    """Distinguish NVIDIA CUDA from Hygon's torch.cuda-compatible HIP runtime."""

    torch_version = str(getattr(torch, "__version__", "")).lower()
    if "metax" in torch_version:
        return "native_metax"

    hip_version = getattr(torch.version, "hip", None)
    if not hip_version:
        return "native_cuda"

    try:
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(device_id).strip().lower()
                if device_name.startswith("bw") or "hygon" in device_name:
                    return "native_hygon"
    except Exception:
        pass

    if _truthy(os.environ.get("GRAPHCAST_ASSUME_HYGON")):
        return "native_hygon"
    return "native_cuda"


def _version(package_names: tuple[str, ...]) -> str:
    for package_name in package_names:
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return "not-installed"


def _module_origin(module_name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return str(Path(spec.origin).resolve())


def _import_torch_npu() -> None:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("NPU requested, but torch_npu is not importable") from exc


def _import_torch_musa() -> None:
    try:
        import torch_musa  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("MUSA requested, but torch_musa is not importable") from exc


def _npu_is_available() -> bool:
    try:
        _import_torch_npu()
    except RuntimeError:
        return False
    return bool(hasattr(torch, "npu") and torch.npu.is_available())


def _musa_is_available() -> bool:
    try:
        _import_torch_musa()
    except RuntimeError:
        return False
    return bool(hasattr(torch, "musa") and torch.musa.is_available())


def _infer_backend_from_device(requested_device: str) -> str:
    if requested_device.startswith("musa"):
        return "native_mthreads"
    if requested_device.startswith("cuda"):
        return _cuda_backend_profile()
    if requested_device.startswith("npu"):
        return "native_npu"
    if requested_device == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return _cuda_backend_profile()
    if _musa_is_available():
        return "native_mthreads"
    if _npu_is_available():
        return "native_npu"
    return "cpu"


def _normalize_backend_name(value: str | None, requested_device: str) -> str:
    raw_name = (value or "").strip().lower()
    if not raw_name:
        return _infer_backend_from_device(requested_device)
    name = BACKEND_ALIASES.get(raw_name, raw_name)
    if name == "auto":
        return _infer_backend_from_device(requested_device)
    if name not in BACKEND_PROFILES:
        allowed = ", ".join(sorted(BACKEND_PROFILES))
        raise ValueError(f"unknown GRAPHCAST_BACKEND={value!r}; allowed: {allowed}")
    return name


def _resolve_device(profile: BackendProfile, requested_device: str) -> torch.device:
    device_value = requested_device if requested_device != "auto" else profile.default_device
    if profile.device_type == "musa":
        _import_torch_musa()
    device = torch.device(device_value)
    if device.type != profile.device_type:
        raise RuntimeError(
            f"backend {profile.name!r} expects device type {profile.device_type!r}, "
            f"but --device resolved to {device!s}"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
    if device.type == "npu":
        _import_torch_npu()
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("NPU requested, but torch.npu.is_available() is false")
    if device.type == "musa" and (not hasattr(torch, "musa") or not torch.musa.is_available()):
        raise RuntimeError("MUSA requested, but torch.musa.is_available() is false")
    return device


def _parse_ops(value: str | None) -> tuple[str, ...]:
    raw_ops = value if value is not None else ",".join(SAFE_FLAGGEMS_OPS)
    return tuple(op.strip() for op in raw_ops.split(",") if op.strip())


def _flagos_mode() -> str:
    mode = os.environ.get("GRAPHCAST_FLAGOS_MODE", "off").strip().lower()
    aliases = {"native": "off", "none": "off", "0": "off", "false": "off"}
    mode = aliases.get(mode, mode)
    if mode not in ("off", "flaggems", "flagtree"):
        raise ValueError(
            "GRAPHCAST_FLAGOS_MODE must be one of off, flaggems, flagtree; "
            f"got {mode!r}"
        )
    return mode


def _default_record_path() -> str:
    return str(Path("outputs/logs") / f"flag_gems_record_{os.getpid()}.log")


def _configure_flagos_toolchain(strict: bool) -> tuple[str | None, tuple[str, ...]]:
    """Apply optional runtime toolchain path overrides for FlagGems/Triton."""

    warnings: list[str] = []
    opt_dir = os.environ.get("GRAPHCAST_BISHENGIR_OPT_DIR")
    if not opt_dir:
        return None, ()

    opt_path = Path(opt_dir) / "bishengir-opt"
    if not opt_path.exists():
        message = f"GRAPHCAST_BISHENGIR_OPT_DIR does not contain bishengir-opt: {opt_dir}"
        if strict:
            raise RuntimeError(message)
        warnings.append(message)
        return opt_dir, tuple(warnings)

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if opt_dir not in path_entries:
        os.environ["PATH"] = opt_dir + os.pathsep + os.environ.get("PATH", "")
    return opt_dir, tuple(warnings)


def _enable_flaggems(profile: BackendProfile, ops: tuple[str, ...], strict: bool) -> FlagOSState:
    warnings: list[str] = []
    if not profile.supports_flaggems:
        message = f"backend {profile.name} does not declare FlagGems support"
        if strict:
            raise RuntimeError(message)
        return FlagOSState("flaggems", strict, ops, "unsupported", warnings=(message,))

    bishengir_opt_dir, toolchain_warnings = _configure_flagos_toolchain(strict)
    warnings.extend(toolchain_warnings)

    risky = [op for op in ops if op in RISKY_FLAGGEMS_OPS]
    if risky:
        warnings.append(
            "risky FlagGems ops requested for GraphCast 25km shapes: " + ",".join(risky)
        )

    vendor = os.environ.get("GEMS_VENDOR") or profile.flaggems_vendor
    if vendor:
        os.environ.setdefault("GEMS_VENDOR", vendor)

    record_path = os.environ.get("GRAPHCAST_FLAGGEMS_RECORD_PATH")
    if not record_path and _truthy(os.environ.get("GRAPHCAST_FLAGGEMS_RECORD")):
        record_path = _default_record_path()

    try:
        import flag_gems

        _bind_ascend_record_loggers()
        patched_ops: list[str] = []
        if _using_ascend_vendor() and "index" in ops:
            # GraphCast gathers node features as src[:, idx, :] (dim-1 index).
            # Ascend FlagGems' index wrapper only handles dim-0 contiguous indices,
            # so we patch it to transpose dim-1 to dim-0 before the wrapper call
            # and restore the original layout afterwards.
            if _patch_ascend_index_for_graphcast(flag_gems):
                patched_ops.append("index.Tensor:graphcast_dim1_gather")

        record_kwargs: dict[str, Any] = {}
        if record_path:
            Path(record_path).parent.mkdir(parents=True, exist_ok=True)
            record_kwargs = {"record": True, "path": record_path}
        if hasattr(flag_gems, "only_enable"):
            flag_gems.only_enable(include=list(ops), **record_kwargs)
            status = "enabled_only_enable"
        elif hasattr(flag_gems, "use_gems"):
            flag_gems.use_gems()
            status = "enabled_use_gems"
            warnings.append("flag_gems.only_enable is unavailable; enabled all FlagGems ops")
        else:
            raise RuntimeError("flag_gems has neither only_enable nor use_gems")

        _bind_ascend_record_loggers()
        registered_ops = _safe_list(flag_gems, "all_registered_ops")
        registered_keys = _safe_list(flag_gems, "all_registered_keys")
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        return FlagOSState(
            "flaggems",
            strict,
            ops,
            "failed",
            vendor=vendor,
            record_path=record_path,
            bishengir_opt_dir=bishengir_opt_dir,
            warnings=tuple(warnings),
            error=f"{type(exc).__name__}: {exc}",
        )

    return FlagOSState(
        "flaggems",
        strict,
        ops,
        status,
        vendor=vendor,
        record_path=record_path,
        bishengir_opt_dir=bishengir_opt_dir,
        warnings=tuple(warnings),
        patched_ops=tuple(patched_ops),
        registered_ops=tuple(registered_ops + registered_keys),
    )


def _configure_flagos(profile: BackendProfile) -> FlagOSState:
    mode = _flagos_mode()
    strict = _truthy(os.environ.get("GRAPHCAST_STRICT_FLAGOS"))
    ops = _parse_ops(os.environ.get("GRAPHCAST_FLAGGEMS_OPS"))
    if mode == "off":
        return FlagOSState(mode, strict, ops, "disabled")
    if mode == "flaggems":
        return _enable_flaggems(profile, ops, strict)

    message = "FlagTree is planned but not wired into the 25km package yet"
    if strict:
        raise RuntimeError(message)
    return FlagOSState(mode, strict, ops, "not_implemented", warnings=(message,))


def _runtime_versions() -> dict[str, str]:
    versions = {
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None) or "not-available",
        "torch_hip": getattr(torch.version, "hip", None) or "not-available",
        "torch_npu": _version(("torch_npu", "torch-npu")),
        "torch_musa": _version(("torch_musa", "torch-musa")),
        "torch_metax": _version(("torch_metax", "torch-metax")),
        "flag_gems": _version(("flag_gems", "FlagGems")),
        "flagtree": _version(("flagtree", "FlagTree")),
    }
    return versions


def resolve_runtime_backend(requested_device: str = "auto") -> RuntimeBackend:
    """Resolve device/backend from CLI device and GRAPHCAST_* environment."""

    backend_name = _normalize_backend_name(
        os.environ.get("GRAPHCAST_BACKEND"),
        requested_device,
    )
    profile = BACKEND_PROFILES[backend_name]
    flagos = _configure_flagos(profile)
    device = _resolve_device(profile, requested_device)
    return RuntimeBackend(
        profile=profile,
        device=device,
        flagos=flagos,
        versions=_runtime_versions(),
    )


def synchronize(runtime: RuntimeBackend | torch.device) -> None:
    device = runtime.device if isinstance(runtime, RuntimeBackend) else runtime
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)
    elif device.type == "musa" and hasattr(torch, "musa"):
        torch.musa.synchronize(device)


def reset_peak_memory(runtime: RuntimeBackend | torch.device) -> None:
    device = runtime.device if isinstance(runtime, RuntimeBackend) else runtime
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.reset_peak_memory_stats(device)
    elif device.type == "musa" and hasattr(torch, "musa"):
        torch.musa.reset_peak_memory_stats(device)


def memory_text(runtime: RuntimeBackend | torch.device) -> str:
    device = runtime.device if isinstance(runtime, RuntimeBackend) else runtime
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        return f"cuda allocated={allocated:.3f} GiB, reserved={reserved:.3f} GiB"
    if device.type == "npu" and hasattr(torch, "npu"):
        allocated = torch.npu.max_memory_allocated(device) / 1024**3
        reserved = torch.npu.max_memory_reserved(device) / 1024**3
        return f"npu allocated={allocated:.3f} GiB, reserved={reserved:.3f} GiB"
    if device.type == "musa" and hasattr(torch, "musa"):
        allocated = torch.musa.max_memory_allocated(device) / 1024**3
        reserved = torch.musa.max_memory_reserved(device) / 1024**3
        return f"musa allocated={allocated:.3f} GiB, reserved={reserved:.3f} GiB"
    return "n/a"


def backend_report_lines(runtime: RuntimeBackend) -> list[str]:
    metadata_dict = runtime.to_metadata()
    return [
        f"- backend: `{metadata_dict['backend']}`",
        f"- backend device type: `{metadata_dict['device_type']}`",
        f"- runtime family: `{metadata_dict['runtime_family']}`",
        f"- chip: `{metadata_dict['chip']}`",
        f"- FlagOS mode: `{metadata_dict['flagos_mode']}`",
        f"- FlagOS status: `{metadata_dict['flagos_status']}`",
        f"- FlagGems ops: `{metadata_dict['flaggems_ops']}`",
        f"- FlagGems vendor: `{metadata_dict['flaggems_vendor']}`",
        f"- FlagGems record path: `{metadata_dict['flaggems_record_path']}`",
        f"- FlagGems source root: `{metadata_dict['flaggems_source_root']}`",
        f"- FlagGems module origin: `{metadata_dict['flaggems_module_origin']}`",
        f"- BishengIR opt dir: `{metadata_dict['bishengir_opt_dir']}`",
        f"- gather backend: `{metadata_dict['gather_backend']}`",
        f"- FlagGems patched ops: `{metadata_dict['flaggems_patched_ops']}`",
        f"- FlagGems registered ops: `{metadata_dict['flaggems_registered_ops']}`",
        f"- runtime versions: `{metadata_dict['versions']}`",
        f"- FlagOS warnings: `{metadata_dict['flagos_warnings']}`",
        f"- FlagOS error: `{metadata_dict['flagos_error']}`",
    ]


def graphcast_gather_mode() -> str:
    return os.environ.get("GRAPHCAST_GATHER_MODE", "index_select").strip().lower()


def use_advanced_index_gather() -> bool:
    return graphcast_gather_mode() in {"advanced_index", "index", "flaggems_index"}


def _using_ascend_vendor() -> bool:
    return os.environ.get("GEMS_VENDOR", "").strip().lower() in {"ascend", "npu"}


def _bind_ascend_record_loggers() -> None:
    flag_logger = logging.getLogger("flag_gems")
    flag_handlers = list(flag_logger.handlers)
    for name in (
        "_ascend.ops.index",
        "_ascend.ops.index_add",
        "flag_gems.ops.index_add",
        "flag_gems.runtime.backend._ascend.ops.index",
        "flag_gems.runtime.backend._ascend.ops.index_add",
        "flag_gems.runtime._ascend.ops.index",
        "flag_gems.runtime._ascend.ops.index_add",
    ):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if name.startswith("flag_gems."):
            logger.propagate = True
            continue
        logger.propagate = False
        for handler in flag_handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)


def _patch_ascend_index_for_graphcast(flag_gems: Any) -> bool:
    try:
        index_module = importlib.import_module(
            "flag_gems.runtime.backend._ascend.ops.index"
        )
    except Exception:
        try:
            index_module = importlib.import_module("flag_gems.runtime._ascend.ops.index")
        except Exception:
            return False

    index_wrapper = getattr(index_module, "index_wrapper", None)
    if index_wrapper is None:
        return False

    original_index = getattr(index_module, "index", None)
    patched_index = _build_graphcast_ascend_index(original_index, index_wrapper)
    setattr(index_module, "index", patched_index)
    return _replace_flaggems_config_function(flag_gems, "index.Tensor", patched_index)


def _build_graphcast_ascend_index(
    original_index: Callable[..., torch.Tensor] | None,
    index_wrapper: Callable[..., None],
) -> Callable[..., torch.Tensor]:
    def graphcast_index(inp: torch.Tensor, indices: tuple[Any, ...]) -> torch.Tensor:
        logging.getLogger("flag_gems.runtime._ascend.ops.index").debug("GEMS_ASCEND INDEX")
        if _is_graphcast_dim1_index(inp, indices):
            index_tensor = indices[1]
            if index_tensor.device != inp.device:
                index_tensor = index_tensor.to(inp.device)
            moved = inp.movedim(1, 0).contiguous()
            out_shape = (index_tensor.numel(), inp.shape[0], *inp.shape[2:])
            out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
            if moved.numel() != 0 and out.numel() != 0:
                index_wrapper(moved, [index_tensor], out)
            return out.movedim(0, 1).contiguous()
        if original_index is None:
            return inp[indices]
        return original_index(inp, indices)

    graphcast_index.__name__ = "index"
    return graphcast_index


def _is_graphcast_dim1_index(inp: torch.Tensor, indices: tuple[Any, ...]) -> bool:
    if inp.ndim < 3 or len(indices) != inp.ndim:
        return False
    if not isinstance(indices[1], torch.Tensor):
        return False
    if indices[1].dtype not in {torch.int64, torch.int32, torch.long, torch.int}:
        return False
    for dim, index in enumerate(indices):
        if dim == 1:
            continue
        if index is None:
            continue
        if isinstance(index, slice) and index == slice(None):
            continue
        if index is not Ellipsis:
            return False
    return True


def _replace_flaggems_config_function(
    flag_gems: Any,
    op_name: str,
    replacement: Callable[..., torch.Tensor],
) -> bool:
    changed = False
    full_config = getattr(flag_gems, "_FULL_CONFIG", None)
    if isinstance(full_config, tuple):
        items = list(full_config)
        for index, item in enumerate(items):
            if item and item[0] == op_name:
                items[index] = (item[0], replacement, *item[2:])
                changed = True
        if changed:
            setattr(flag_gems, "_FULL_CONFIG", tuple(items))
    by_func = getattr(flag_gems, "FULL_CONFIG_BY_FUNC", None)
    if isinstance(by_func, dict):
        for key, values in list(by_func.items()):
            if not isinstance(values, list):
                continue
            new_values = []
            for item in values:
                if item and item[0] == op_name:
                    new_values.append((item[0], replacement, *item[2:]))
                    changed = True
                else:
                    new_values.append(item)
            by_func[key] = new_values
        by_func["index"] = [
            (item[0], replacement, *item[2:])
            for item in (getattr(flag_gems, "_FULL_CONFIG", ()) or ())
            if item and item[0] == op_name
        ]
    return changed


def _safe_list(module: Any, attr: str) -> list[str]:
    try:
        value = getattr(module, attr)()
    except Exception:
        return []
    return [str(item) for item in value]
