#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
-------
One-click runner for your DFEM/PINN project (PyCharm 直接运行即可).

本版包含：
- 启用 TF 显存分配器 cuda_malloc_async（需在 import TF 之前设置）
- 自动解析 INP & 表面 key（支持精确/模糊；含 bolt2 的 ASM::"bolt2 uo"）
- 与新版 surfaces.py / inp_parser.py 对齐（ELEMENT 表面可直接采样）
- 训练配置集中覆盖（降显存：节点前向分块、降低采样规模、混合精度）
- 训练配置由 config.yaml 驱动（未找到或缺失必填项会直接报错）
- 训练结束后在 outputs/ 生成随机 5 组镜面变形云图（文件名含螺母拧紧角度）
"""

# ====== 必须在导入 TensorFlow 之前设置 ======
import os
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")  # 可选：减少冗余日志
# ============================================

import sys
import re
import atexit
import argparse
import json
import math
import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import yaml  # 新增：读取 config.yaml

# --- 确保 "src" 在 Python 路径中 ---
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

CONFIG_PATH = os.path.join(ROOT, "config.yaml")

_LOG_READY = False
_LOG_FILES = []
_LOG_STDOUT = None
_LOG_STDERR = None
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class _Tee:
    def __init__(self, *streams, filters=None):
        self._streams = streams
        if filters is None:
            filters = [None] * len(streams)
        if len(filters) < len(streams):
            filters = list(filters) + [None] * (len(streams) - len(filters))
        self._filters = filters

    def write(self, data):
        if not isinstance(data, str):
            data = str(data)
        for stream, filt in zip(self._streams, self._filters):
            out = data if filt is None else filt(data)
            if out:
                stream.write(out)
        for stream in self._streams:
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def __getattr__(self, name):
        return getattr(self._streams[0], name)


def _strip_ansi(text: str) -> str:
    text = _ANSI_RE.sub("", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _setup_run_logs(log_dir: str = "", prefix: str = "train"):
    """Duplicate stdout/stderr to files while keeping console output."""
    global _LOG_READY, _LOG_FILES, _LOG_STDOUT, _LOG_STDERR
    if _LOG_READY:
        return
    base = log_dir or ROOT
    os.makedirs(base, exist_ok=True)
    stdout_path = os.path.join(base, f"{prefix}.log")
    stderr_path = os.path.join(base, f"{prefix}.err")
    stdout_f = open(stdout_path, "w", encoding="utf-8-sig", buffering=1)
    stderr_f = open(stderr_path, "w", encoding="utf-8-sig", buffering=1)
    _LOG_FILES = [stdout_f, stderr_f]
    _LOG_STDOUT = sys.stdout
    _LOG_STDERR = sys.stderr
    sys.stdout = _Tee(_LOG_STDOUT, stdout_f, filters=[None, _strip_ansi])
    sys.stderr = _Tee(_LOG_STDERR, stderr_f, filters=[None, _strip_ansi])
    _LOG_READY = True

    def _close_logs():
        global _LOG_READY, _LOG_FILES, _LOG_STDOUT, _LOG_STDERR
        if _LOG_STDOUT is not None:
            sys.stdout = _LOG_STDOUT
        if _LOG_STDERR is not None:
            sys.stderr = _LOG_STDERR
        for handle in _LOG_FILES:
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass
        _LOG_FILES = []
        _LOG_STDOUT = None
        _LOG_STDERR = None
        _LOG_READY = False

    atexit.register(_close_logs)

# ---------- SavedModel 默认输出路径 ----------
def _default_saved_model_dir(out_dir: str) -> str:
    """Return a timestamped SavedModel export directory under ``out_dir``."""

    base = os.path.abspath(out_dir or "outputs")
    os.makedirs(base, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(base, f"saved_model_{ts}")


@dataclass
class _TrainingPhaseResult:
    phase_name: str
    trainer: object
    best_ckpt_path: str = ""
    final_ckpt_path: str = ""
    out_dir: str = ""
    ckpt_dir: str = ""
    export_dir: str = ""
    summary_path: str = ""


def _get_meta_value(payload, key: str, default=""):
    if payload is None:
        return default
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _to_json_scalar(value):
    if value is None:
        return None
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except Exception:
            return None
    if isinstance(value, (list, tuple, set, dict)):
        return None
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, int):
            return value
        return float(value)
    except Exception:
        return None


def build_paper_benchmark_summary(benchmark_meta, phase_result):
    trainer = getattr(phase_result, "trainer", None)
    trainer_snapshot = {}
    if trainer is not None and hasattr(trainer, "get_compact_final_metrics_snapshot"):
        trainer_snapshot = trainer.get_compact_final_metrics_snapshot() or {}

    identity = {
        "family_id": str(_get_meta_value(benchmark_meta, "family_id", "") or ""),
        "variant_id": str(_get_meta_value(benchmark_meta, "variant_id", "") or ""),
        "family_label": str(_get_meta_value(benchmark_meta, "family_label", "") or ""),
        "variant_label": str(_get_meta_value(benchmark_meta, "variant_label", "") or ""),
        "run_id": str(_get_meta_value(benchmark_meta, "run_id", "") or ""),
        "config_path": str(_get_meta_value(benchmark_meta, "config_path", "") or ""),
        "results_dir": str(
            _get_meta_value(benchmark_meta, "results_dir", "") or getattr(phase_result, "out_dir", "") or ""
        ),
        "phase_name": str(getattr(phase_result, "phase_name", "") or ""),
    }

    cfg = getattr(trainer, "cfg", None)
    mixed_phase = getattr(cfg, "mixed_bilevel_phase", None)
    route = {
        "training_profile": str(getattr(cfg, "training_profile", "") or ""),
        "phase_name": str(getattr(mixed_phase, "phase_name", "") or ""),
        "route_mode": str(trainer_snapshot.get("route_mode", "") or ""),
        "normal_ift_enabled": bool(getattr(mixed_phase, "normal_ift_enabled", False)),
        "tangential_ift_enabled": bool(getattr(mixed_phase, "tangential_ift_enabled", False)),
        "detach_inner_solution": bool(getattr(mixed_phase, "detach_inner_solution", False)),
        "allow_full_ift_warmstart": bool(getattr(mixed_phase, "allow_full_ift_warmstart", False)),
        "coupling_tightening_protocol": str(getattr(cfg, "coupling_tightening_protocol", "") or ""),
        "normal_ift_ready": _to_json_scalar(trainer_snapshot.get("normal_ift_ready")),
        "normal_ift_consumed": _to_json_scalar(trainer_snapshot.get("normal_ift_consumed")),
        "strict_effective_traction_scale": _to_json_scalar(
            trainer_snapshot.get("strict_effective_traction_scale")
        ),
        "coupling_phase_traction_scale": _to_json_scalar(
            trainer_snapshot.get("coupling_phase_traction_scale")
        ),
        "coupling_refinement_steps": _to_json_scalar(
            trainer_snapshot.get("coupling_refinement_steps")
        ),
        "coupling_tail_qn_budget": _to_json_scalar(
            trainer_snapshot.get("coupling_tail_qn_budget")
        ),
    }

    outcome = {
        "out_dir": str(getattr(phase_result, "out_dir", "") or ""),
        "ckpt_dir": str(getattr(phase_result, "ckpt_dir", "") or ""),
        "export_dir": str(getattr(phase_result, "export_dir", "") or ""),
        "best_ckpt_path": str(getattr(phase_result, "best_ckpt_path", "") or ""),
        "final_ckpt_path": str(getattr(phase_result, "final_ckpt_path", "") or ""),
        "mean_Pi": _to_json_scalar(trainer_snapshot.get("mean_Pi")),
        "mean_E_data": _to_json_scalar(trainer_snapshot.get("mean_E_data")),
        "mean_ft_residual_norm": _to_json_scalar(trainer_snapshot.get("mean_ft_residual_norm")),
        "mean_inner_convergence_rate": _to_json_scalar(
            trainer_snapshot.get("mean_inner_convergence_rate")
        ),
        "mean_inner_fallback_rate": _to_json_scalar(
            trainer_snapshot.get("mean_inner_fallback_rate")
        ),
        "final_metrics": {
            key: _to_json_scalar(value)
            for key, value in (trainer_snapshot or {}).items()
        },
    }

    return {
        "identity": identity,
        "route": route,
        "outcome": outcome,
    }


def write_paper_benchmark_summary(summary, summary_path):
    path = os.fspath(summary_path)
    if not path:
        raise ValueError("summary_path must be provided")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return Path(path)


def _resolve_paper_benchmark_summary_path(cfg) -> str:
    candidate = (
        getattr(cfg, "paper_benchmark_summary_path", "")
        or getattr(cfg, "benchmark_summary_path", "")
        or getattr(getattr(cfg, "output_config", None), "paper_benchmark_summary_path", "")
        or getattr(getattr(cfg, "output_config", None), "benchmark_summary_path", "")
    )
    return os.fspath(candidate).strip() if candidate else ""


def _derive_phase_config(base_cfg: "TrainerConfig", phase_name: str) -> "TrainerConfig":
    """Clone the parsed trainer config and apply one two-stage phase override."""

    phase_key = str(phase_name or "").strip().lower()
    if phase_key not in {"phase1", "phase2"}:
        raise ValueError(f"Unsupported two-stage phase: {phase_name!r}")

    cfg = copy.deepcopy(base_cfg)
    phase_override = getattr(cfg.two_stage_training, phase_key)

    if phase_override.max_steps is not None:
        cfg.max_steps = int(phase_override.max_steps)
        cfg.adam_steps = int(phase_override.max_steps)
    if phase_override.lr is not None:
        cfg.lr = float(phase_override.lr)
    if phase_override.save_best_on is not None:
        cfg.save_best_on = str(phase_override.save_best_on)
    if phase_override.validation_eval_every is not None:
        cfg.validation_eval_every = int(phase_override.validation_eval_every)
    if phase_override.supervision_contribution_floor_ratio is not None:
        cfg.supervision_contribution_floor_ratio = float(phase_override.supervision_contribution_floor_ratio)
    if phase_override.resume_ckpt_path:
        cfg.resume_ckpt_path = str(phase_override.resume_ckpt_path)
    if int(getattr(phase_override, "resume_start_step", 0) or 0) > 0:
        cfg.resume_start_step = int(phase_override.resume_start_step)
    if phase_override.resume_reset_optimizer is not None:
        cfg.resume_reset_optimizer = bool(phase_override.resume_reset_optimizer)

    phase_weight_map = {
        "w_int": "w_int",
        "w_cn": "w_cn",
        "w_ct": "w_ct",
        "w_bc": "w_bc",
        "w_tight": "w_tight",
        "w_sigma": "w_sigma",
        "w_eq": "w_eq",
        "w_reg": "w_reg",
        "w_data": "w_data",
        "w_delta_data": "w_delta_data",
        "w_optical_modal": "w_optical_modal",
        "w_smooth": "w_smooth",
    }
    for key, value in (phase_override.base_weights or {}).items():
        attr = phase_weight_map.get(str(key).strip())
        if attr is None:
            continue
        setattr(cfg.total_cfg, attr, float(value))
    mixed_phase_override = getattr(phase_override, "mixed_bilevel_phase", {}) or {}
    if isinstance(mixed_phase_override, dict) and mixed_phase_override:
        if "phase_name" in mixed_phase_override:
            cfg.mixed_bilevel_phase.phase_name = str(mixed_phase_override["phase_name"])
        if "normal_ift_enabled" in mixed_phase_override:
            cfg.mixed_bilevel_phase.normal_ift_enabled = bool(mixed_phase_override["normal_ift_enabled"])
        if "tangential_ift_enabled" in mixed_phase_override:
            cfg.mixed_bilevel_phase.tangential_ift_enabled = bool(mixed_phase_override["tangential_ift_enabled"])
        if "detach_inner_solution" in mixed_phase_override:
            cfg.mixed_bilevel_phase.detach_inner_solution = bool(mixed_phase_override["detach_inner_solution"])
        if "allow_full_ift_warmstart" in mixed_phase_override:
            cfg.mixed_bilevel_phase.allow_full_ift_warmstart = bool(
                mixed_phase_override["allow_full_ift_warmstart"]
            )

    cfg.run_phase_name = phase_key
    cfg.out_dir = os.path.join(base_cfg.out_dir, phase_key)
    cfg.ckpt_dir = os.path.join(base_cfg.ckpt_dir, phase_key)
    return cfg


def _allocate_run_checkpoint_dir(base_ckpt_dir: str) -> str:
    """Create a per-run checkpoint directory under the requested phase root."""

    root_dir = base_ckpt_dir or "checkpoints"
    ts_tag = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    candidate = os.path.join(root_dir, ts_tag)
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(root_dir, f"{ts_tag}-{suffix}")
        suffix += 1
    os.makedirs(candidate, exist_ok=True)
    return candidate


def _resolve_export_dir(cfg: "TrainerConfig", export_saved_model):
    """Resolve the SavedModel export directory for one training phase."""

    if export_saved_model is None:
        return ""

    export_dir = str(export_saved_model or "").strip()
    if export_dir:
        export_dir = os.path.abspath(export_dir)
        parent_dir = os.path.dirname(export_dir)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        return export_dir

    export_dir = _default_saved_model_dir(cfg.out_dir)
    print(f"[main] 鏈彁渚?--export锛屽皢 SavedModel 鍐欏叆: {export_dir}")
    return export_dir

# --- 项目内模块导入 ---
def _resolve_export_dir(cfg: "TrainerConfig", export_saved_model):
    """Resolve the SavedModel export directory for one training phase."""

    if export_saved_model is None:
        return ""

    export_dir = str(export_saved_model or "").strip()
    if export_dir:
        export_dir = os.path.abspath(export_dir)
        parent_dir = os.path.dirname(export_dir)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        return export_dir

    export_dir = _default_saved_model_dir(cfg.out_dir)
    print(f"[main] --export not provided; writing SavedModel to: {export_dir}")
    return export_dir


from train.trainer import TrainerConfig
from inp_io.cdb_parser import load_cdb
from mesh.contact_pairs import guess_surface_key
from physics.physical_scales import PhysicalScaleConfig

_LOCKED_ROUTE_NAME = "force_then_lock+incremental"
_LOCKED_TRAINING_PROFILE = "locked"
_STRICT_MIXED_EXPERIMENTAL_PROFILE = "strict_mixed_experimental"
_STRICT_MIXED_POST_REENTRY_PROFILE = "strict_mixed_experimental_post_reentry"
_NORMAL_CONTACT_FIRST_MAINLINE_PROFILE = "normal_contact_first_mainline"
_NORMAL_CONTACT_TIGHTENING_PROTOCOL = "progressive_normal_contact"
_NORMAL_CONTACT_TIGHTENING_STAGE_LABELS = (
    "weak_coupling_warmup",
    "transition_normal_coupling",
    "strict_normal_coupling",
)
_P3_LEARNING_GATE_PROFILE = "p3_learning_gate"
_P5A_REENTRY_GATE_PROFILE = "p5a_reentry_gate"
_P5B1_PHYSICS_REENTRY_GATE_PROFILE = "p5b1_physics_reentry_gate"
_VALID_TRAINING_PROFILES = frozenset(
    {
        _LOCKED_TRAINING_PROFILE,
        _STRICT_MIXED_EXPERIMENTAL_PROFILE,
        _STRICT_MIXED_POST_REENTRY_PROFILE,
        _NORMAL_CONTACT_FIRST_MAINLINE_PROFILE,
        _P3_LEARNING_GATE_PROFILE,
        _P5A_REENTRY_GATE_PROFILE,
        _P5B1_PHYSICS_REENTRY_GATE_PROFILE,
    }
)


def _enforce_locked_route(cfg: TrainerConfig) -> None:
    """Enforce the single supported training route."""

    issues = []
    stage_mode = str(getattr(cfg.total_cfg, "preload_stage_mode", "") or "")
    stage_mode = stage_mode.strip().lower().replace("-", "_")

    if not bool(getattr(cfg, "preload_use_stages", False)):
        issues.append("preload_use_stages must be true")
    if not bool(getattr(cfg, "incremental_mode", False)):
        issues.append("incremental_mode must be true")
    if stage_mode != "force_then_lock":
        issues.append("preload_stage_mode must be force_then_lock")
    if bool(getattr(cfg, "stage_resample_contact", False)):
        issues.append("stage_resample_contact must be false")
    if int(getattr(cfg, "resample_contact_every", 0) or 0) > 0:
        issues.append("resample_contact_every must be <= 0")
    if bool(getattr(cfg, "contact_rar_enabled", False)):
        issues.append("contact_rar_enabled must be false")
    if bool(getattr(cfg, "volume_rar_enabled", False)):
        issues.append("volume_rar_enabled must be false")
    if bool(getattr(cfg, "lbfgs_enabled", False)):
        issues.append("optimizer_config.lbfgs.enabled must be false")
    if bool(getattr(cfg, "friction_smooth_schedule", False)):
        issues.append("friction_config.smooth_to_strict must be false")
    if bool(getattr(cfg, "viz_compare_cases", False)):
        issues.append("output_config.viz_compare_cases must be false")

    if issues:
        joined = "; ".join(issues)
        raise ValueError(f"Locked route {_LOCKED_ROUTE_NAME} violation: {joined}")

    # Canonicalize implied values for the locked route.
    cfg.alm_update_every = 0
    cfg.contact_cfg.update_every_steps = 1


def _canonicalize_locked_route(cfg: TrainerConfig) -> None:
    """Force canonical values for the only supported training route."""

    cfg.preload_use_stages = True
    cfg.incremental_mode = True
    cfg.preload_randomize_order = False
    cfg.total_cfg.preload_stage_mode = "force_then_lock"

    # Locked-route implied runtime cadence.
    cfg.alm_update_every = 0
    cfg.contact_cfg.update_every_steps = 1
    cfg.elas_cfg.cache_sample_metrics = False


def _normalize_training_profile(raw_profile) -> str:
    profile = str(raw_profile or _LOCKED_TRAINING_PROFILE).strip().lower().replace("-", "_")
    if not profile:
        profile = _LOCKED_TRAINING_PROFILE
    if profile not in _VALID_TRAINING_PROFILES:
        valid = ", ".join(sorted(_VALID_TRAINING_PROFILES))
        raise ValueError(f"Unsupported training_profile '{profile}'. Expected one of: {valid}.")
    return profile


def _resolve_training_profile(cfg_yaml, config_path: str) -> str:
    raw_profile = None
    if isinstance(cfg_yaml, dict):
        raw_profile = cfg_yaml.get("training_profile", None)
    profile = _normalize_training_profile(raw_profile)
    if raw_profile is None:
        base_name = os.path.splitext(os.path.basename(str(config_path or "")))[0].strip().lower().replace("-", "_")
        if base_name == _STRICT_MIXED_EXPERIMENTAL_PROFILE:
            profile = _STRICT_MIXED_EXPERIMENTAL_PROFILE
        elif base_name == _STRICT_MIXED_POST_REENTRY_PROFILE:
            profile = _STRICT_MIXED_POST_REENTRY_PROFILE
        elif base_name == _NORMAL_CONTACT_FIRST_MAINLINE_PROFILE:
            profile = _NORMAL_CONTACT_FIRST_MAINLINE_PROFILE
        elif base_name == "strict_mixed_p3_learning_gate":
            profile = _P3_LEARNING_GATE_PROFILE
        elif base_name == "strict_mixed_p5a_reentry_gate":
            profile = _P5A_REENTRY_GATE_PROFILE
        elif base_name == "strict_mixed_p5b1_physics_reentry_gate":
            profile = _P5B1_PHYSICS_REENTRY_GATE_PROFILE
    return profile


def _validate_experimental_profile(cfg: TrainerConfig) -> None:
    profile_name = str(getattr(cfg, "training_profile", _STRICT_MIXED_EXPERIMENTAL_PROFILE) or _STRICT_MIXED_EXPERIMENTAL_PROFILE)
    phase_name = str(getattr(getattr(cfg, "mixed_bilevel_phase", None), "phase_name", "") or "").strip().lower()
    if phase_name in {"", "phase0"}:
        raise ValueError(
            f"training_profile='{profile_name}' requires mixed_bilevel_phase.phase_name to be non-phase0."
        )
    backend = str(getattr(cfg, "contact_backend", "auto") or "auto").strip().lower()
    if backend not in {"auto", "inner_solver"}:
        raise ValueError(
            f"training_profile='{profile_name}' only supports contact_backend 'auto' or 'inner_solver'."
        )


def _validate_normal_contact_first_mainline_profile(cfg: TrainerConfig) -> None:
    profile_name = str(
        getattr(cfg, "training_profile", _NORMAL_CONTACT_FIRST_MAINLINE_PROFILE)
        or _NORMAL_CONTACT_FIRST_MAINLINE_PROFILE
    )
    phase_name = str(getattr(getattr(cfg, "mixed_bilevel_phase", None), "phase_name", "") or "").strip().lower()
    if phase_name in {"", "phase0"}:
        raise ValueError(
            f"training_profile='{profile_name}' requires mixed_bilevel_phase.phase_name to be non-phase0."
        )
    if not bool(getattr(cfg.mixed_bilevel_phase, "normal_ift_enabled", False)):
        raise ValueError(
            f"training_profile='{profile_name}' requires mixed_bilevel_phase.normal_ift_enabled=true."
        )
    if bool(getattr(cfg.mixed_bilevel_phase, "tangential_ift_enabled", False)):
        raise ValueError(
            f"training_profile='{profile_name}' requires mixed_bilevel_phase.tangential_ift_enabled=false."
        )
    if bool(getattr(cfg.mixed_bilevel_phase, "detach_inner_solution", True)):
        raise ValueError(
            f"training_profile='{profile_name}' requires mixed_bilevel_phase.detach_inner_solution=false."
        )
    backend = str(getattr(cfg, "contact_backend", "auto") or "auto").strip().lower()
    if backend not in {"auto", "inner_solver"}:
        raise ValueError(
            f"training_profile='{profile_name}' only supports contact_backend 'auto' or 'inner_solver'."
        )


def _canonicalize_normal_contact_first_mainline(cfg: TrainerConfig) -> None:
    protocol = str(
        getattr(cfg, "coupling_tightening_protocol", _NORMAL_CONTACT_TIGHTENING_PROTOCOL)
        or _NORMAL_CONTACT_TIGHTENING_PROTOCOL
    ).strip().lower().replace("-", "_")
    if not protocol:
        protocol = _NORMAL_CONTACT_TIGHTENING_PROTOCOL
    cfg.coupling_tightening_protocol = protocol

    raw_labels = getattr(cfg, "coupling_tightening_stage_labels", _NORMAL_CONTACT_TIGHTENING_STAGE_LABELS)
    labels = tuple(str(x or "").strip() for x in (raw_labels or ()))
    if len(labels) != 3 or any(not label for label in labels):
        labels = _NORMAL_CONTACT_TIGHTENING_STAGE_LABELS
    cfg.coupling_tightening_stage_labels = labels


def _canonicalize_p3_learning_gate(cfg: TrainerConfig) -> None:
    cfg.risk_guard_enabled = False
    cfg.risk_guard_scale = 1.0
    cfg.protect_prefix_enabled = False
    cfg.protect_first_n_steps = 0
    cfg.guard_activate_after_first_c_event = False
    cfg.contact_backend = "auto"
    cfg.max_tail_qn_iters = 0
    cfg.max_inner_iters_signature_gate = ""
    cfg.signature_gated_max_inner_iters = 0
    cfg.mixed_bilevel_phase.phase_name = "phase0"
    cfg.mixed_bilevel_phase.normal_ift_enabled = False
    cfg.mixed_bilevel_phase.tangential_ift_enabled = False
    cfg.mixed_bilevel_phase.detach_inner_solution = True
    cfg.total_cfg.w_int = 0.0
    cfg.total_cfg.w_cn = 0.0
    cfg.total_cfg.w_ct = 0.0
    cfg.total_cfg.w_bc = 0.0
    cfg.total_cfg.w_tight = 0.0
    cfg.total_cfg.w_sigma = 0.0
    cfg.total_cfg.w_eq = 0.0
    cfg.total_cfg.w_bi = 0.0
    cfg.total_cfg.w_ed = 0.0
    cfg.uncertainty_loss_weight = 0.0


def _canonicalize_p5a_reentry_gate(cfg: TrainerConfig) -> None:
    _canonicalize_p3_learning_gate(cfg)


def _canonicalize_p5b1_physics_reentry_gate(cfg: TrainerConfig) -> None:
    w_bc = float(getattr(cfg.total_cfg, "w_bc", 0.0) or 0.0)
    w_eq = float(getattr(cfg.total_cfg, "w_eq", 0.0) or 0.0)
    _canonicalize_p5a_reentry_gate(cfg)
    cfg.total_cfg.w_bc = w_bc if w_bc > 0.0 else 1.0
    cfg.total_cfg.w_eq = w_eq if w_eq > 0.0 else 1.0


def _validate_p3_learning_gate_profile(cfg: TrainerConfig) -> None:
    sup_cfg = getattr(cfg, "supervision", None)
    if sup_cfg is None or not bool(getattr(sup_cfg, "enabled", False)):
        raise ValueError("training_profile='p3_learning_gate' requires supervision.enabled=true.")
    if not str(getattr(sup_cfg, "single_case_id", "") or "").strip():
        raise ValueError("training_profile='p3_learning_gate' requires supervision.single_case_id.")
    stages = tuple(int(x) for x in (getattr(sup_cfg, "single_case_stages", ()) or ()))
    if not stages:
        raise ValueError("training_profile='p3_learning_gate' requires supervision.single_case_stages.")
    feature_mode = str(getattr(sup_cfg, "feature_mode", "cartesian") or "cartesian").strip().lower()
    target_frame = str(getattr(sup_cfg, "target_frame", "cartesian") or "cartesian").strip().lower()
    if feature_mode not in {"cartesian", "plain_input", "ring_aware"}:
        raise ValueError(f"Unsupported supervision.feature_mode for p3_learning_gate: {feature_mode!r}")
    if target_frame not in {"cartesian", "cylindrical"}:
        raise ValueError(f"Unsupported supervision.target_frame for p3_learning_gate: {target_frame!r}")
    if feature_mode == "ring_aware" or target_frame == "cylindrical":
        center = getattr(sup_cfg, "annulus_center", None)
        if not isinstance(center, tuple) or len(center) != 2:
            raise ValueError("training_profile='p3_learning_gate' requires supervision.annulus_center for annulus routes.")
        r_in = getattr(sup_cfg, "annulus_r_in", None)
        r_out = getattr(sup_cfg, "annulus_r_out", None)
        if r_in is None or r_out is None:
            raise ValueError("training_profile='p3_learning_gate' requires annulus_r_in/annulus_r_out for annulus routes.")
        if float(r_out) <= float(r_in):
            raise ValueError("training_profile='p3_learning_gate' requires annulus_r_out > annulus_r_in.")
        if int(getattr(sup_cfg, "annulus_fourier_order", 0) or 0) <= 0:
            raise ValueError("training_profile='p3_learning_gate' requires annulus_fourier_order > 0.")


def _validate_p5a_reentry_gate_profile(cfg: TrainerConfig) -> None:
    sup_cfg = getattr(cfg, "supervision", None)
    if sup_cfg is None or not bool(getattr(sup_cfg, "enabled", False)):
        raise ValueError("training_profile='p5a_reentry_gate' requires supervision.enabled=true.")
    selected_case_ids = tuple(str(x) for x in (getattr(sup_cfg, "selected_case_ids", ()) or ()))
    if len(selected_case_ids) != 6:
        raise ValueError(
            "training_profile='p5a_reentry_gate' requires supervision.selected_case_ids with exactly 6 cases."
        )
    stages = tuple(int(x) for x in (getattr(sup_cfg, "single_case_stages", ()) or ()))
    if stages != (1, 2, 3):
        raise ValueError("training_profile='p5a_reentry_gate' requires supervision.single_case_stages = (1, 2, 3).")
    feature_mode = str(getattr(sup_cfg, "feature_mode", "cartesian") or "cartesian").strip().lower()
    target_frame = str(getattr(sup_cfg, "target_frame", "cartesian") or "cartesian").strip().lower()
    if feature_mode not in {"cartesian", "plain_input"}:
        raise ValueError("training_profile='p5a_reentry_gate' requires supervision.feature_mode='cartesian'.")
    if target_frame != "cylindrical":
        raise ValueError("training_profile='p5a_reentry_gate' requires supervision.target_frame='cylindrical'.")
    center = getattr(sup_cfg, "annulus_center", None)
    if not isinstance(center, tuple) or len(center) != 2:
        raise ValueError("training_profile='p5a_reentry_gate' requires supervision.annulus_center.")
    r_in = getattr(sup_cfg, "annulus_r_in", None)
    r_out = getattr(sup_cfg, "annulus_r_out", None)
    if r_in is None or r_out is None:
        raise ValueError("training_profile='p5a_reentry_gate' requires annulus_r_in/annulus_r_out.")
    if float(r_out) <= float(r_in):
        raise ValueError("training_profile='p5a_reentry_gate' requires annulus_r_out > annulus_r_in.")
    if int(getattr(sup_cfg, "annulus_fourier_order", 0) or 0) <= 0:
        raise ValueError("training_profile='p5a_reentry_gate' requires annulus_fourier_order > 0.")
    field_cfg = getattr(getattr(cfg, "model_cfg", None), "field", None)
    if field_cfg is None:
        raise ValueError("training_profile='p5a_reentry_gate' requires model_cfg.field.")
    if not bool(getattr(field_cfg, "internal_ring_lift_enabled", False)):
        raise ValueError("training_profile='p5a_reentry_gate' requires network_config.internal_ring_lift_enabled=true.")
    if not bool(getattr(field_cfg, "cylindrical_primary_head_enabled", False)):
        raise ValueError(
            "training_profile='p5a_reentry_gate' requires network_config.cylindrical_primary_head_enabled=true."
        )
    internal_center = tuple(getattr(field_cfg, "internal_ring_center", ()) or ())
    if len(internal_center) != 2:
        raise ValueError("training_profile='p5a_reentry_gate' requires network_config.internal_ring_center.")
    if float(getattr(field_cfg, "internal_ring_r_out", 0.0) or 0.0) <= float(
        getattr(field_cfg, "internal_ring_r_in", 0.0) or 0.0
    ):
        raise ValueError(
            "training_profile='p5a_reentry_gate' requires network_config.internal_ring_r_out > internal_ring_r_in."
        )
    if int(getattr(field_cfg, "internal_ring_fourier_order", 0) or 0) <= 0:
        raise ValueError(
            "training_profile='p5a_reentry_gate' requires network_config.internal_ring_fourier_order > 0."
        )


def _validate_p5b1_physics_reentry_gate_profile(cfg: TrainerConfig) -> None:
    _validate_p5a_reentry_gate_profile(cfg)
    if float(getattr(cfg.total_cfg, "w_eq", 0.0) or 0.0) <= 0.0:
        raise ValueError("training_profile='p5b1_physics_reentry_gate' requires loss_config.base_weights.w_eq > 0.")
    if float(getattr(cfg.total_cfg, "w_bc", 0.0) or 0.0) <= 0.0:
        raise ValueError("training_profile='p5b1_physics_reentry_gate' requires loss_config.base_weights.w_bc > 0.")


# ---------- 工具：读取 config.yaml（容错） ----------
def _load_yaml_config(config_path=None):
    path = str(config_path).strip() if config_path else CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到配置文件（路径: {path}），请先准备配置文件后再运行。")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    print(f"[main] 成功读取 {os.path.basename(path)}。")
    return data


# ---------- 小工具：容错匹配表面 key ----------
def _auto_resolve_surface_keys(asm, key_or_hint: str) -> str:
    """
    支持“精确 key 或模糊片段”的自动匹配。
    - 若 key_or_hint 正好是 asm.surfaces 的键，直接返回；
    - 否则进行大小写不敏感的包含匹配；唯一匹配则返回该 key；否则抛出错误提示。
    """
    k = key_or_hint
    if k in asm.surfaces:
        return k
    g = guess_surface_key(asm, k)
    if g is not None:
        return g
    low = k.strip().lower()
    cands = [kk for kk, s in asm.surfaces.items()
             if low in kk.lower() or low in s.name.strip().lower()]
    if len(cands) == 1:
        return cands[0]
    elif len(cands) == 0:
        raise KeyError(f"找不到包含 '{k}' 的表面；请在 config.yaml 或 main.py 里把名字改得更准确一些。")
    else:
        msg = "匹配到多个表面：\n  " + "\n  ".join(cands) + "\n请改成更精确的名字。"
        raise KeyError(msg)


# ---------- 读取 CDB + 组装 TrainerConfig（并返回 asm 以供审计打印） ----------
def _prepare_config_with_autoguess(config_path=None):
    config_path = str(config_path).strip() if config_path else CONFIG_PATH
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)

    def _resolve_optional_path(raw_val):
        if raw_val is None:
            return None
        txt = str(raw_val).strip()
        if not txt:
            return None
        if os.path.isabs(txt):
            return txt
        candidates = [
            os.path.join(config_dir, txt),
            os.path.join(ROOT, txt),
            os.path.join(os.path.dirname(ROOT), txt),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return os.path.abspath(cand)
        return os.path.abspath(os.path.join(ROOT, txt))

    # 0) 读取 config.yaml（若存在）
    cfg_yaml = _load_yaml_config(config_path)

    # 1) 模型路径 (inp / cdb)
    inp_path = (
        cfg_yaml.get("inp_path", "")
        or cfg_yaml.get("cdb_path", "")
        or cfg_yaml.get("mesh_path", "")
    ).strip()
    if not inp_path:
        raise ValueError("config.yaml 必须提供 inp_path/cdb_path/mesh_path。")
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"未找到网格文件：{inp_path}。请在 config.yaml 中填写正确路径。")
    ext = os.path.splitext(inp_path)[1].lower()
    if ext == ".cdb":
        asm = load_cdb(inp_path)
    elif ext == ".inp":
        from inp_io.inp_parser import load_inp
        asm = load_inp(inp_path)
    else:
        raise ValueError("当前仅支持 .cdb 或 .inp 网格文件")

    # 2) 镜面表面名
    mirror_surface_name = cfg_yaml.get("mirror_surface_name", "NONE").strip()
    try:
        _ = _auto_resolve_surface_keys(asm, mirror_surface_name)
    except Exception as e:
        print("[main] 提示：镜面表面名自动匹配失败：", e)
        print("       继续使用你提供的名字（可视化时按该名字模糊匹配）。")

    # 3) 螺母拧紧：优先读取 nuts；否则自动探测 LUOMU* 部件
    nut_specs = []
    for b in cfg_yaml.get("nuts", []) or []:
        nut_specs.append(
            {
                "name": b.get("name", ""),
                "part": b.get("part", b.get("part_name", "")),
                "axis": b.get("axis", None),
                "center": b.get("center", None),
            }
        )

    if not nut_specs:
        for pname in getattr(asm, "parts", {}).keys():
            if "LUOMU" in pname.upper():
                nut_specs.append({"name": pname, "part": pname})
        if nut_specs:
            print(f"[main] 自动识别螺母部件: {[d['part'] for d in nut_specs]}")
        else:
            print("[main] 未发现螺母部件（LUOMU*），将跳过拧紧约束。")

    # 4) 接触对
    contact_pairs_cfg = cfg_yaml.get("contact_pairs", []) or []

    contact_pairs = []
    if contact_pairs_cfg:
        for p in contact_pairs_cfg:
            try:
                slave_key = _auto_resolve_surface_keys(asm, p["slave_key"])
                master_key = _auto_resolve_surface_keys(asm, p["master_key"])
                contact_pairs.append(
                    {
                        "slave_key": slave_key,
                        "master_key": master_key,
                        "name": p.get("name", ""),
                        "interaction": p.get("interaction", ""),
                    }
                )
            except Exception as e:
                print(f"[main] 接触对 '{p.get('name','')}' 自动匹配失败：{e}")
                print("       暂时跳过该接触对（可在 config.yaml 的 contact_pairs 中修正后再跑）。")

    # 5) 材料与 Part→材料映射
    mat_props = cfg_yaml.get("material_properties", {}) or {}
    if not isinstance(mat_props, dict) or not mat_props:
        raise ValueError("config.yaml 必须提供非空的 material_properties。")
    materials = {}
    yield_candidates = []
    for name, props in mat_props.items():
        E = props.get("E", None)
        nu = props.get("nu", None)
        if E is None or nu is None:
            continue
        E_f = float(E)
        nu_f = float(nu)
        if E_f <= 0.0:
            raise ValueError(f"material_properties.{name}.E 必须为正值，当前为 {E}")
        if not (-1.0 < nu_f < 0.5):
            raise ValueError(f"material_properties.{name}.nu 超出物理范围 (-1,0.5)，当前为 {nu}")
        if nu_f > 0.495:
            print(f"[main] 警告：材料 {name} 的 ν={nu_f} 接近 0.5，线弹性可能病态。")
        if E_f < 1e2 or E_f > 1e7:
            print(f"[main] 警告：材料 {name} 的 E={E_f:g} 量级异常，请确认单位是否为 MPa。")
        materials[name] = (E_f, nu_f)

        # 收集屈服强度候选（若提供）
        for k in ("sigma_y_tension", "sigma_y_compression", "sigma_y", "yield_strength"):
            v = props.get(k, None)
            if v is None:
                continue
            try:
                yield_candidates.append(float(v))
            except Exception:
                pass
    if not materials:
        raise ValueError("material_properties 解析后为空，请检查配置内容。")

    part2mat = cfg_yaml.get("part2mat", {}) or {}
    if not part2mat:
        raise ValueError("config.yaml 必须提供非空的 part2mat。")

    # 6) 训练步数与采样设置：优先使用 config.yaml 中的 optimizer_config / elasticity_config
    optimizer_cfg = cfg_yaml.get("optimizer_config", {}) or {}
    elas_cfg_yaml = cfg_yaml.get("elasticity_config", {}) or {}

    train_steps = int(optimizer_cfg.get("epochs", TrainerConfig.max_steps))
    n_contact_points_per_pair = int(cfg_yaml.get("n_contact_points_per_pair", TrainerConfig.n_contact_points_per_pair))
    contact_two_pass = bool(cfg_yaml.get("contact_two_pass", TrainerConfig.contact_two_pass))
    contact_mode = str(cfg_yaml.get("contact_mode", TrainerConfig.contact_mode))
    contact_mortar_gauss = int(cfg_yaml.get("contact_mortar_gauss", TrainerConfig.contact_mortar_gauss))
    contact_mortar_max_points = int(cfg_yaml.get("contact_mortar_max_points", TrainerConfig.contact_mortar_max_points))
    contact_mode_norm = contact_mode.strip().lower()
    if contact_mode_norm not in {"sample", "mortar"}:
        raise ValueError(f"contact_mode 仅支持 'sample' 或 'mortar'，当前为 {contact_mode!r}")
    if contact_mode_norm == "mortar":
        if contact_mortar_gauss not in (1, 3, 7):
            raise ValueError(f"contact_mortar_gauss 仅支持 1/3/7，当前为 {contact_mortar_gauss}")
        if contact_two_pass:
            print("[main] contact_mode=mortar 时忽略 contact_two_pass（仅使用单向 mortar 接触）。")
            contact_two_pass = False
    preload_face_points_each = int(
        cfg_yaml.get("tightening_n_points_each", cfg_yaml.get("preload_n_points_each", TrainerConfig.preload_n_points_each))
    )
    preload_min = cfg_yaml.get("tighten_angle_min", cfg_yaml.get("preload_min", None))
    preload_max = cfg_yaml.get("tighten_angle_max", cfg_yaml.get("preload_max", None))
    preload_range = cfg_yaml.get("tighten_angle_range", cfg_yaml.get("preload_range_n", None))
    if not nut_specs:
        preload_min, preload_max = 0.0, 0.0
    else:
        if preload_min is None or preload_max is None:
            if preload_range is None:
                raise ValueError("config.yaml 必须显式提供 tighten_angle_min/max 或 tighten_angle_range。")
            preload_min, preload_max = float(preload_range[0]), float(preload_range[1])
        else:
            preload_min, preload_max = float(preload_min), float(preload_max)

    # 7) 组装训练配置
    cfg = TrainerConfig(
        inp_path=inp_path,
        mirror_surface_name=mirror_surface_name,  # 可视化仍支持模糊匹配
        materials=materials,
        part2mat=part2mat,
        contact_pairs=contact_pairs,
        n_contact_points_per_pair=n_contact_points_per_pair,
        contact_two_pass=contact_two_pass,
        contact_mode=contact_mode,
        contact_mortar_gauss=contact_mortar_gauss,
        contact_mortar_max_points=contact_mortar_max_points,
        preload_specs=nut_specs,
        preload_n_points_each=preload_face_points_each,
        preload_min=preload_min,
        preload_max=preload_max,
        max_steps=train_steps,
        viz_samples_after_train=5,   # 随机 5 组，标题包含螺母拧紧角度
    )
    cfg.training_profile = _resolve_training_profile(cfg_yaml, config_path)
    if "max_tail_qn_iters" in cfg_yaml:
        cfg.max_tail_qn_iters = max(0, int(cfg_yaml["max_tail_qn_iters"]))
    # Keep the promoted strict-mixed candidate surface minimal and explicit.
    if "tangential_training_mode" in cfg_yaml:
        cfg.tangential_training_mode = str(cfg_yaml["tangential_training_mode"])
    if "risk_guard_enabled" in cfg_yaml:
        cfg.risk_guard_enabled = bool(cfg_yaml["risk_guard_enabled"])
    if "risk_guard_scale" in cfg_yaml:
        cfg.risk_guard_scale = float(cfg_yaml["risk_guard_scale"])
    if "risk_guard_allowed_buckets" in cfg_yaml:
        raw_buckets = cfg_yaml["risk_guard_allowed_buckets"] or []
        if isinstance(raw_buckets, str):
            raw_buckets = [raw_buckets]
        cfg.risk_guard_allowed_buckets = tuple(
            str(bucket).strip().upper() for bucket in raw_buckets if str(bucket).strip()
        )
    if "protect_prefix_enabled" in cfg_yaml:
        cfg.protect_prefix_enabled = bool(cfg_yaml["protect_prefix_enabled"])
    if "protect_first_n_steps" in cfg_yaml:
        cfg.protect_first_n_steps = max(0, int(cfg_yaml["protect_first_n_steps"]))
    scale_cfg_yaml = cfg_yaml.get("physical_scales", {}) or {}
    cfg.physical_scales = PhysicalScaleConfig(
        L_ref=float(scale_cfg_yaml.get("L_ref", cfg_yaml.get("L_ref", cfg.physical_scales.L_ref))),
        u_ref=float(scale_cfg_yaml.get("u_ref", cfg_yaml.get("u_ref", cfg.physical_scales.u_ref))),
        sigma_ref=float(scale_cfg_yaml.get("sigma_ref", cfg_yaml.get("sigma_ref", cfg.total_cfg.sigma_ref))),
        E_ref=float(scale_cfg_yaml.get("E_ref", cfg_yaml.get("E_ref", cfg.physical_scales.E_ref))),
        F_ref=float(scale_cfg_yaml.get("F_ref", cfg_yaml.get("F_ref", cfg.physical_scales.F_ref))),
        A_ref=float(scale_cfg_yaml.get("A_ref", cfg_yaml.get("A_ref", cfg.physical_scales.A_ref))),
    )
    # 若 config.yaml 中提供了材料屈服强度，则默认取最小值作为全局屈服参考
    if yield_candidates:
        try:
            cfg.yield_strength = float(min(yield_candidates))
            print(f"[main] 读取材料屈服强度（最小值）: σy={cfg.yield_strength:g}")
            # 用屈服强度作为应力监督的归一化尺度，使 E_sigma 无量纲且量级稳定
            if (
                cfg.yield_strength
                and cfg.yield_strength > 0
                and float(getattr(cfg.physical_scales, "sigma_ref", 0.0)) <= 0.0
            ):
                cfg.physical_scales.sigma_ref = float(cfg.yield_strength)
        except Exception:
            pass
    cfg.total_cfg.sigma_ref = float(cfg.physical_scales.resolved_sigma_ref())
    print(f"[main] 应力监督归一化参考: sigma_ref={cfg.total_cfg.sigma_ref:g}")
    output_cfg = cfg_yaml.get("output_config", {}) or {}
    if "save_path" in output_cfg:
        cfg.out_dir = str(output_cfg["save_path"])

    tight_cfg = cfg_yaml.get("tightening_config", {}) or {}
    if "alpha" in tight_cfg:
        cfg.tightening_cfg.alpha = float(tight_cfg["alpha"])
    if "angle_unit" in tight_cfg:
        cfg.tightening_cfg.angle_unit = str(tight_cfg["angle_unit"])
    if "clockwise" in tight_cfg:
        cfg.tightening_cfg.clockwise = bool(tight_cfg["clockwise"])
    if "forward_chunk" in tight_cfg:
        cfg.tightening_cfg.forward_chunk = int(tight_cfg["forward_chunk"])

    # Mixed precision: default to fp32 unless explicitly enabled in config.yaml
    if "mixed_precision" in cfg_yaml:
        mp_cfg = cfg_yaml.get("mixed_precision", None)
        if mp_cfg is None or mp_cfg is False:
            cfg.mixed_precision = None
        elif isinstance(mp_cfg, bool) and mp_cfg is True:
            cfg.mixed_precision = "mixed_float16"
        else:
            cfg.mixed_precision = str(mp_cfg)

    cfg.viz_use_shape_function_interp = bool(
        output_cfg.get("viz_use_shape_function_interp", cfg.viz_use_shape_function_interp)
    )
    if "viz_use_last_training_case" in output_cfg:
        cfg.viz_use_last_training_case = bool(output_cfg["viz_use_last_training_case"])
    if "viz_samples_after_train" in output_cfg:
        cfg.viz_samples_after_train = int(output_cfg["viz_samples_after_train"])
    if "viz_plot_stages" in output_cfg:
        cfg.viz_plot_stages = bool(output_cfg["viz_plot_stages"])
    if "viz_skip_release_stage_plot" in output_cfg:
        cfg.viz_skip_release_stage_plot = bool(output_cfg["viz_skip_release_stage_plot"])
    if "viz_write_reference_aligned" in output_cfg:
        cfg.viz_write_reference_aligned = bool(output_cfg["viz_write_reference_aligned"])
    for key in (
        "viz_reference_truth_path",
        "viz_truth_path",
        "truth_data_path",
        "truth_path",
    ):
        if key in output_cfg:
            val = output_cfg.get(key)
            cfg.viz_reference_truth_path = None if val is None else str(val)
            break
    if "viz_surface_source" in output_cfg:
        cfg.viz_surface_source = str(output_cfg["viz_surface_source"])
    if "viz_refine_subdivisions" in output_cfg:
        cfg.viz_refine_subdivisions = int(output_cfg["viz_refine_subdivisions"])
    if "viz_smooth_vector_iters" in output_cfg:
        cfg.viz_smooth_vector_iters = int(output_cfg["viz_smooth_vector_iters"])
    if "viz_smooth_vector_lambda" in output_cfg:
        cfg.viz_smooth_vector_lambda = float(output_cfg["viz_smooth_vector_lambda"])
    if "viz_smooth_scalar_iters" in output_cfg:
        cfg.viz_smooth_scalar_iters = int(output_cfg["viz_smooth_scalar_iters"])
    if "viz_smooth_scalar_lambda" in output_cfg:
        cfg.viz_smooth_scalar_lambda = float(output_cfg["viz_smooth_scalar_lambda"])
    if "viz_eval_scope" in output_cfg:
        cfg.viz_eval_scope = str(output_cfg["viz_eval_scope"])
    if "viz_eval_batch_size" in output_cfg:
        cfg.viz_eval_batch_size = int(output_cfg["viz_eval_batch_size"])
    if "viz_force_pointwise" in output_cfg:
        cfg.viz_force_pointwise = bool(output_cfg["viz_force_pointwise"])
    if "viz_supervision_compare_enabled" in output_cfg:
        cfg.viz_supervision_compare_enabled = bool(output_cfg["viz_supervision_compare_enabled"])
    if "viz_supervision_compare_split" in output_cfg:
        cfg.viz_supervision_compare_split = str(output_cfg["viz_supervision_compare_split"])
    if "viz_supervision_compare_sources" in output_cfg:
        raw = output_cfg.get("viz_supervision_compare_sources") or ()
        if isinstance(raw, str):
            raw = [x.strip() for x in raw.split(",")]
        cfg.viz_supervision_compare_sources = tuple(str(x).strip() for x in raw if str(x).strip())
    if "viz_same_pipeline_supervision_debug" in output_cfg:
        cfg.viz_same_pipeline_supervision_debug = bool(output_cfg["viz_same_pipeline_supervision_debug"])
    if "viz_export_final_and_best" in output_cfg:
        cfg.viz_export_final_and_best = bool(output_cfg["viz_export_final_and_best"])
    if "save_best_on" in output_cfg:
        cfg.save_best_on = str(output_cfg["save_best_on"])
    cfg.adam_steps = cfg.max_steps

    cfg.lr = float(optimizer_cfg.get("learning_rate", cfg.lr))
    if "resume_ckpt_path" in cfg_yaml:
        cfg.resume_ckpt_path = str(cfg_yaml["resume_ckpt_path"] or "").strip() or None
    if "resume_start_step" in cfg_yaml:
        cfg.resume_start_step = max(0, int(cfg_yaml["resume_start_step"] or 0))
    if "resume_reset_optimizer" in cfg_yaml:
        cfg.resume_reset_optimizer = bool(cfg_yaml["resume_reset_optimizer"])
    if "grad_clip_norm" in optimizer_cfg:
        cfg.grad_clip_norm = float(optimizer_cfg["grad_clip_norm"])
    if "log_every" in optimizer_cfg:
        cfg.log_every = int(optimizer_cfg["log_every"])
    if "validation_eval_every" in optimizer_cfg:
        cfg.validation_eval_every = int(optimizer_cfg["validation_eval_every"])
    if "contact_route_update_every" in cfg_yaml:
        cfg.contact_route_update_every = int(cfg_yaml["contact_route_update_every"])
    lr_decay_cfg = optimizer_cfg.get("lr_decay_on_plateau", {}) or {}
    if isinstance(lr_decay_cfg, dict):
        if "enabled" in lr_decay_cfg:
            cfg.val_plateau_lr_decay_enabled = bool(lr_decay_cfg["enabled"])
        if "metric" in lr_decay_cfg:
            cfg.val_plateau_lr_decay_metric = str(lr_decay_cfg["metric"])
        if "warmup_steps" in lr_decay_cfg:
            cfg.val_plateau_lr_decay_warmup = int(lr_decay_cfg["warmup_steps"])
        if "patience" in lr_decay_cfg:
            cfg.val_plateau_lr_decay_patience = int(lr_decay_cfg["patience"])
        if "factor" in lr_decay_cfg:
            cfg.val_plateau_lr_decay_factor = float(lr_decay_cfg["factor"])
        if "min_lr" in lr_decay_cfg:
            cfg.val_plateau_lr_decay_min_lr = float(lr_decay_cfg["min_lr"])
    early_exit_cfg = optimizer_cfg.get("early_exit", None)
    if not isinstance(early_exit_cfg, dict):
        early_exit_cfg = cfg_yaml.get("early_exit", {}) or {}
    else:
        early_exit_cfg = early_exit_cfg or {}
    if "enabled" in early_exit_cfg:
        cfg.early_exit_enabled = bool(early_exit_cfg["enabled"])
    if "warmup_steps" in early_exit_cfg:
        cfg.early_exit_warmup_steps = int(early_exit_cfg["warmup_steps"])
    if "nonfinite_patience" in early_exit_cfg:
        cfg.early_exit_nonfinite_patience = int(early_exit_cfg["nonfinite_patience"])
    if "divergence_patience" in early_exit_cfg:
        cfg.early_exit_divergence_patience = int(early_exit_cfg["divergence_patience"])
    if "grad_norm_threshold" in early_exit_cfg:
        cfg.early_exit_grad_norm_threshold = float(early_exit_cfg["grad_norm_threshold"])
    if "pi_ema_rel_increase" in early_exit_cfg:
        cfg.early_exit_pi_ema_rel_increase = float(early_exit_cfg["pi_ema_rel_increase"])
    if "check_every" in early_exit_cfg:
        cfg.early_exit_check_every = int(early_exit_cfg["check_every"])
    elif "early_exit_check_every" in cfg_yaml:
        cfg.early_exit_check_every = int(cfg_yaml["early_exit_check_every"])

    two_stage_cfg = cfg_yaml.get("two_stage_training", {}) or {}
    if isinstance(two_stage_cfg, dict) and two_stage_cfg:
        if "enabled" in two_stage_cfg:
            cfg.two_stage_training.enabled = bool(two_stage_cfg["enabled"])

        def _apply_two_stage_phase_overrides(phase_cfg, phase_yaml):
            if not isinstance(phase_yaml, dict):
                return
            if "max_steps" in phase_yaml:
                phase_cfg.max_steps = int(phase_yaml["max_steps"])
            if "learning_rate" in phase_yaml:
                phase_cfg.lr = float(phase_yaml["learning_rate"])
            if "save_best_on" in phase_yaml:
                phase_cfg.save_best_on = str(phase_yaml["save_best_on"])
            if "validation_eval_every" in phase_yaml:
                phase_cfg.validation_eval_every = int(phase_yaml["validation_eval_every"])
            if "supervision_contribution_floor_ratio" in phase_yaml:
                phase_cfg.supervision_contribution_floor_ratio = float(
                    phase_yaml["supervision_contribution_floor_ratio"]
                )
            if "resume_ckpt_path" in phase_yaml:
                phase_cfg.resume_ckpt_path = str(phase_yaml["resume_ckpt_path"] or "").strip() or None
            if "resume_start_step" in phase_yaml:
                phase_cfg.resume_start_step = max(0, int(phase_yaml["resume_start_step"] or 0))
            if "resume_reset_optimizer" in phase_yaml:
                phase_cfg.resume_reset_optimizer = bool(phase_yaml["resume_reset_optimizer"])
            mixed_phase = phase_yaml.get("mixed_bilevel_phase", {}) or {}
            if isinstance(mixed_phase, dict) and mixed_phase:
                parsed_mixed_phase = {}
                if "phase_name" in mixed_phase:
                    parsed_mixed_phase["phase_name"] = str(mixed_phase["phase_name"])
                if "normal_ift_enabled" in mixed_phase:
                    parsed_mixed_phase["normal_ift_enabled"] = bool(mixed_phase["normal_ift_enabled"])
                if "tangential_ift_enabled" in mixed_phase:
                    parsed_mixed_phase["tangential_ift_enabled"] = bool(mixed_phase["tangential_ift_enabled"])
                if "detach_inner_solution" in mixed_phase:
                    parsed_mixed_phase["detach_inner_solution"] = bool(mixed_phase["detach_inner_solution"])
                if "allow_full_ift_warmstart" in mixed_phase:
                    parsed_mixed_phase["allow_full_ift_warmstart"] = bool(mixed_phase["allow_full_ift_warmstart"])
                phase_cfg.mixed_bilevel_phase = parsed_mixed_phase
            base_weights = phase_yaml.get("base_weights", phase_yaml.get("loss_weights", {})) or {}
            if isinstance(base_weights, dict) and base_weights:
                phase_cfg.base_weights = {str(k): float(v) for k, v in base_weights.items()}

        _apply_two_stage_phase_overrides(cfg.two_stage_training.phase1, two_stage_cfg.get("phase1", {}) or {})
        _apply_two_stage_phase_overrides(cfg.two_stage_training.phase2, two_stage_cfg.get("phase2", {}) or {})

    # ===== 拧紧分阶段 / 顺序设置 =====
    staging_cfg = cfg_yaml.get("preload_staging", {}) or {}
    stage_mode_top = cfg_yaml.get("preload_stage_mode", None)
    if stage_mode_top is not None:
        cfg.total_cfg.preload_stage_mode = str(stage_mode_top)
    if "mode" in staging_cfg:
        cfg.total_cfg.preload_stage_mode = str(staging_cfg["mode"])

    # 顶层布尔开关优先，其次是 staging_cfg 内的 enabled
    use_stages_val = cfg_yaml.get("preload_use_stages", None)
    if use_stages_val is not None:
        cfg.preload_use_stages = bool(use_stages_val)
    if "enabled" in staging_cfg:
        cfg.preload_use_stages = bool(staging_cfg["enabled"])

    random_order_val = cfg_yaml.get("preload_randomize_order", None)
    if random_order_val is not None:
        cfg.preload_randomize_order = bool(random_order_val)
    if "randomize_order" in staging_cfg:
        cfg.preload_randomize_order = bool(staging_cfg["randomize_order"])
    append_release_stage_val = cfg_yaml.get("preload_append_release_stage", None)
    if append_release_stage_val is None and "append_release_stage" in staging_cfg:
        append_release_stage_val = staging_cfg["append_release_stage"]
    append_release_stage_explicit = append_release_stage_val is not None
    if append_release_stage_explicit:
        cfg.preload_append_release_stage = bool(append_release_stage_val)

    if "repeat" in staging_cfg:
        cfg.preload_sequence_repeat = int(staging_cfg["repeat"])
    if "shuffle" in staging_cfg:
        cfg.preload_sequence_shuffle = bool(staging_cfg["shuffle"])
    if "jitter" in staging_cfg:
        cfg.preload_sequence_jitter = float(staging_cfg["jitter"])

    seq_overrides = cfg_yaml.get("preload_sequence", None)
    if seq_overrides:
        cfg.preload_sequence = list(seq_overrides)
    seq_from_staging = staging_cfg.get("sequence", None)
    if seq_from_staging:
        cfg.preload_sequence = list(seq_from_staging)

    if cfg.preload_sequence:
        cfg.preload_use_stages = True

    supervision_cfg = cfg_yaml.get("supervision", {}) or {}
    if isinstance(supervision_cfg, dict) and supervision_cfg:
        if "enabled" in supervision_cfg:
            cfg.supervision.enabled = bool(supervision_cfg["enabled"])
        else:
            cfg.supervision.enabled = True
        if "case_table_path" in supervision_cfg:
            cfg.supervision.case_table_path = _resolve_optional_path(supervision_cfg["case_table_path"])
        if "stage_dir" in supervision_cfg:
            cfg.supervision.stage_dir = _resolve_optional_path(supervision_cfg["stage_dir"])
        if "stage_count" in supervision_cfg:
            cfg.supervision.stage_count = int(supervision_cfg["stage_count"])
        if "single_case_id" in supervision_cfg:
            raw = supervision_cfg.get("single_case_id")
            cfg.supervision.single_case_id = None if raw is None else str(raw)
        if "selected_case_ids" in supervision_cfg:
            raw = supervision_cfg.get("selected_case_ids") or []
            if isinstance(raw, (str, int, float)):
                raw = [raw]
            cfg.supervision.selected_case_ids = tuple(str(x) for x in raw)
        if "single_case_stages" in supervision_cfg:
            raw = supervision_cfg.get("single_case_stages") or []
            if isinstance(raw, (int, float, str)):
                raw = [raw]
            cfg.supervision.single_case_stages = tuple(int(x) for x in raw)
        if "feature_mode" in supervision_cfg:
            cfg.supervision.feature_mode = str(supervision_cfg["feature_mode"])
        if "target_frame" in supervision_cfg:
            cfg.supervision.target_frame = str(supervision_cfg["target_frame"])
        if "annulus_center" in supervision_cfg:
            raw = supervision_cfg.get("annulus_center") or ()
            if len(raw) != 2:
                raise ValueError("supervision.annulus_center must contain exactly two values.")
            cfg.supervision.annulus_center = (float(raw[0]), float(raw[1]))
        if "annulus_r_in" in supervision_cfg:
            cfg.supervision.annulus_r_in = float(supervision_cfg["annulus_r_in"])
        if "annulus_r_out" in supervision_cfg:
            cfg.supervision.annulus_r_out = float(supervision_cfg["annulus_r_out"])
        if "annulus_fourier_order" in supervision_cfg:
            cfg.supervision.annulus_fourier_order = int(supervision_cfg["annulus_fourier_order"])
        if "shuffle" in supervision_cfg:
            cfg.supervision.shuffle = bool(supervision_cfg["shuffle"])
        if "seed" in supervision_cfg:
            cfg.supervision.seed = int(supervision_cfg["seed"])
        if "split_group_key" in supervision_cfg:
            cfg.supervision.split_group_key = str(supervision_cfg["split_group_key"])
        if "split_stratify_key" in supervision_cfg:
            raw = supervision_cfg["split_stratify_key"]
            cfg.supervision.split_stratify_key = None if raw is None else str(raw)
        if "test_group_quotas" in supervision_cfg:
            raw = supervision_cfg.get("test_group_quotas") or {}
            cfg.supervision.test_group_quotas = {str(k): int(v) for k, v in dict(raw).items()}
        if "cv_n_folds" in supervision_cfg:
            cfg.supervision.cv_n_folds = int(supervision_cfg["cv_n_folds"])
        if "cv_fold_index" in supervision_cfg:
            cfg.supervision.cv_fold_index = int(supervision_cfg["cv_fold_index"])
        if "train_splits" in supervision_cfg:
            raw = supervision_cfg.get("train_splits") or []
            if isinstance(raw, str):
                raw = [raw]
            cfg.supervision.train_splits = tuple(str(x) for x in raw)
        if "eval_splits" in supervision_cfg:
            raw = supervision_cfg.get("eval_splits") or []
            if isinstance(raw, str):
                raw = [raw]
            cfg.supervision.eval_splits = tuple(str(x) for x in raw)
        if "export_eval_reports" in supervision_cfg:
            cfg.supervision.export_eval_reports = bool(supervision_cfg["export_eval_reports"])
        if "export_eval_plots" in supervision_cfg:
            cfg.supervision.export_eval_plots = bool(supervision_cfg["export_eval_plots"])
        if cfg.supervision.enabled and not append_release_stage_explicit:
            n_bolts = max(1, len(cfg.preload_specs) or 3)
            cfg.preload_append_release_stage = int(cfg.supervision.stage_count) > n_bolts

    # ===== Incremental Mode A (per-stage backprop) =====
    if "incremental_mode" in cfg_yaml:
        cfg.incremental_mode = bool(cfg_yaml.get("incremental_mode"))
    if "stage_inner_steps" in cfg_yaml:
        cfg.stage_inner_steps = int(cfg_yaml.get("stage_inner_steps", cfg.stage_inner_steps))
    if "stage_alm_every" in cfg_yaml:
        cfg.stage_alm_every = int(cfg_yaml.get("stage_alm_every", cfg.stage_alm_every))
    if "reset_contact_state_per_case" in cfg_yaml:
        cfg.reset_contact_state_per_case = bool(cfg_yaml.get("reset_contact_state_per_case"))
    if "stage_schedule_steps" in cfg_yaml:
        schedule = cfg_yaml.get("stage_schedule_steps") or []
        if isinstance(schedule, (list, tuple)):
            cfg.stage_schedule_steps = [int(x) for x in schedule]
    if "coupling_tightening_protocol" in cfg_yaml:
        cfg.coupling_tightening_protocol = str(
            cfg_yaml.get("coupling_tightening_protocol", _NORMAL_CONTACT_TIGHTENING_PROTOCOL)
        )
    if "coupling_tightening_stage_labels" in cfg_yaml:
        labels = cfg_yaml.get("coupling_tightening_stage_labels") or []
        if isinstance(labels, (list, tuple)):
            cfg.coupling_tightening_stage_labels = tuple(str(x) for x in labels)

    # ===== 损失加权配置（含自适应） =====
    mixed_phase_cfg = cfg_yaml.get("mixed_bilevel_phase", {}) or {}
    if isinstance(mixed_phase_cfg, dict) and mixed_phase_cfg:
        if "phase_name" in mixed_phase_cfg:
            cfg.mixed_bilevel_phase.phase_name = str(mixed_phase_cfg["phase_name"])
        if "normal_ift_enabled" in mixed_phase_cfg:
            cfg.mixed_bilevel_phase.normal_ift_enabled = bool(mixed_phase_cfg["normal_ift_enabled"])
        if "tangential_ift_enabled" in mixed_phase_cfg:
            cfg.mixed_bilevel_phase.tangential_ift_enabled = bool(mixed_phase_cfg["tangential_ift_enabled"])
        if "detach_inner_solution" in mixed_phase_cfg:
            cfg.mixed_bilevel_phase.detach_inner_solution = bool(mixed_phase_cfg["detach_inner_solution"])
        if "allow_full_ift_warmstart" in mixed_phase_cfg:
            cfg.mixed_bilevel_phase.allow_full_ift_warmstart = bool(
                mixed_phase_cfg["allow_full_ift_warmstart"]
            )

    if "contact_backend" in cfg_yaml:
        cfg.contact_backend = str(cfg_yaml["contact_backend"])
    if "trainable_scope" in cfg_yaml:
        cfg.trainable_scope = str(cfg_yaml["trainable_scope"])

    continuation_caps_cfg = cfg_yaml.get("continuation_caps", {}) or {}
    if isinstance(continuation_caps_cfg, dict) and continuation_caps_cfg:
        if "eps_shrink" in continuation_caps_cfg:
            cfg.continuation_eps_shrink_cap = float(continuation_caps_cfg["eps_shrink"])
        if "kt_growth" in continuation_caps_cfg:
            cfg.continuation_kt_growth_cap = float(continuation_caps_cfg["kt_growth"])

    loss_cfg_yaml = cfg_yaml.get("loss_config", {}) or {}
    loss_mode = loss_cfg_yaml.get("mode", None)
    if loss_mode is None:
        loss_mode = loss_cfg_yaml.get("loss_mode", None)
    if loss_mode is not None:
        cfg.total_cfg.loss_mode = str(loss_mode)
    base_weights_yaml = loss_cfg_yaml.get("base_weights", {}) or {}
    weight_key_map = {
        "w_int": ("w_int", "E_int"),
        "w_cn": ("w_cn", "E_cn"),
        "w_ct": ("w_ct", "E_ct"),
        "w_bi": ("w_bi", "E_bi"),
        "w_bc": ("w_bc", "E_bc"),
        "w_tight": ("w_tight", "E_tight"),
        "w_sigma": ("w_sigma", "E_sigma"),
        "w_eq": ("w_eq", "E_eq"),
        "w_reg": ("w_reg", "E_reg"),
        "w_ed": ("w_ed", "E_ed"),
        "w_unc": ("w_unc", "E_unc"),
        "w_data": ("w_data", "E_data"),
        "w_delta_data": ("w_delta_data", "E_delta_data"),
        "w_optical_modal": ("w_optical_modal", "E_optical_modal"),
        "w_smooth": ("w_smooth", "E_smooth"),
        "w_path": ("path_penalty_weight", "path_penalty_total"),
        "w_fric_path": ("fric_path_penalty_weight", "fric_path_penalty_total"),
    }
    for yaml_key, (attr, _) in weight_key_map.items():
        if yaml_key in base_weights_yaml:
            setattr(cfg.total_cfg, attr, float(base_weights_yaml[yaml_key]))

    # CRITICAL: Read network_config for DFEM parameters
    net_cfg_yaml = cfg_yaml.get("network_config", {})
    enc_cfg_yaml = net_cfg_yaml.get("condition_encoder", {}) or {}
    explicit_condition_encoder_mode = False
    if isinstance(enc_cfg_yaml, dict) and enc_cfg_yaml:
        if "mode" in enc_cfg_yaml:
            cfg.model_cfg.encoder.mode = str(enc_cfg_yaml["mode"])
            explicit_condition_encoder_mode = True
        if "token_dim" in enc_cfg_yaml:
            cfg.model_cfg.encoder.structured_token_dim = int(enc_cfg_yaml["token_dim"])
        if "token_depth" in enc_cfg_yaml:
            cfg.model_cfg.encoder.structured_token_depth = int(enc_cfg_yaml["token_depth"])
        if "pool" in enc_cfg_yaml:
            cfg.model_cfg.encoder.structured_pool = str(enc_cfg_yaml["pool"])
        if "recency_temperature" in enc_cfg_yaml:
            cfg.model_cfg.encoder.structured_recency_temperature = float(
                enc_cfg_yaml["recency_temperature"]
            )
        if "residual_scale" in enc_cfg_yaml:
            cfg.model_cfg.encoder.structured_residual_scale = float(enc_cfg_yaml["residual_scale"])
        if "state_residual_scale" in enc_cfg_yaml:
            cfg.model_cfg.encoder.structured_residual_scale = float(enc_cfg_yaml["state_residual_scale"])
        if "warmup_steps" in enc_cfg_yaml:
            cfg.ase_residual_warmup_steps = int(enc_cfg_yaml["warmup_steps"])
        if "residual_warmup_steps" in enc_cfg_yaml:
            cfg.ase_residual_warmup_steps = int(enc_cfg_yaml["residual_warmup_steps"])
        encoder_mode_for_role = str(
            getattr(cfg.model_cfg.encoder, "mode", "flat") or "flat"
        ).strip().lower()
        cfg.condition_encoder_role = {
            "structured_bolt_tokens": "supporting_preload_path",
            "assembly_state_evolution": "assembly_state_evolution_path",
        }.get(encoder_mode_for_role, "baseline_flat_condition_path")
        print(
            "[main] Supporting condition encoder:"
            f" mode={cfg.model_cfg.encoder.mode}"
            f" token_dim={cfg.model_cfg.encoder.structured_token_dim}"
            f" token_depth={cfg.model_cfg.encoder.structured_token_depth}"
            f" pool={cfg.model_cfg.encoder.structured_pool}"
            f" recency_temperature={cfg.model_cfg.encoder.structured_recency_temperature:g}"
            f" residual_scale={cfg.model_cfg.encoder.structured_residual_scale:g}"
            f" warmup_steps={cfg.ase_residual_warmup_steps}"
        )
    if not explicit_condition_encoder_mode:
        encoder_mode = str(getattr(cfg.model_cfg.encoder, "mode", "flat") or "flat").strip().lower()
        cfg.condition_encoder_role = {
            "structured_bolt_tokens": "supporting_preload_path",
            "assembly_state_evolution": "assembly_state_evolution_path",
        }.get(encoder_mode, "baseline_flat_condition_path")
        if bool(getattr(cfg, "preload_use_stages", False)) and bool(getattr(cfg, "incremental_mode", False)):
            print("[main] Staged preload keeps baseline flat condition path unless condition_encoder.mode is explicit.")
    elif not hasattr(cfg, "condition_encoder_role"):
        cfg.condition_encoder_role = "baseline_flat_condition_path"
    if str(getattr(cfg, "trainable_scope", "all") or "all").strip().lower() != "all":
        print(f"[main] Trainable scope: {cfg.trainable_scope}")
    if "dfem_mode" in net_cfg_yaml:
        cfg.model_cfg.field.dfem_mode = bool(net_cfg_yaml["dfem_mode"])
        print(f"[main] DFEM mode set from config: {cfg.model_cfg.field.dfem_mode}")
    if "node_emb_dim" in net_cfg_yaml:
        cfg.model_cfg.field.node_emb_dim = int(net_cfg_yaml["node_emb_dim"])
        print(f"[main] Node embedding dim: {cfg.model_cfg.field.node_emb_dim}")
    if "graph_precompute" in net_cfg_yaml:
        cfg.model_cfg.field.graph_precompute = bool(net_cfg_yaml["graph_precompute"])
        print(f"[main] Graph precompute: {cfg.model_cfg.field.graph_precompute}")
    if "graph_k" in net_cfg_yaml:
        cfg.model_cfg.field.graph_k = int(net_cfg_yaml["graph_k"])
        print(f"[main] Graph k: {cfg.model_cfg.field.graph_k}")
    if "graph_width" in net_cfg_yaml:
        cfg.model_cfg.field.graph_width = int(net_cfg_yaml["graph_width"])
        print(f"[main] Graph width: {cfg.model_cfg.field.graph_width}")
    if "graph_layers" in net_cfg_yaml:
        cfg.model_cfg.field.graph_layers = int(net_cfg_yaml["graph_layers"])
        print(f"[main] Graph layers: {cfg.model_cfg.field.graph_layers}")
    if "stress_out_dim" in net_cfg_yaml:
        cfg.model_cfg.field.stress_out_dim = int(net_cfg_yaml["stress_out_dim"])
        print(f"[main] Stress head out dim: {cfg.model_cfg.field.stress_out_dim}")
    if "output_scale" in net_cfg_yaml:
        cfg.model_cfg.field.output_scale = float(net_cfg_yaml["output_scale"])
        print(f"[main] Displacement output scale: {cfg.model_cfg.field.output_scale:g}")
    if "output_scale_trainable" in net_cfg_yaml:
        cfg.model_cfg.field.output_scale_trainable = bool(net_cfg_yaml["output_scale_trainable"])
        print(f"[main] Output scale trainable: {cfg.model_cfg.field.output_scale_trainable}")
    if "use_finite_spectral" in net_cfg_yaml:
        cfg.model_cfg.field.use_finite_spectral = bool(net_cfg_yaml["use_finite_spectral"])
        print(f"[main] Finite spectral encoding: {cfg.model_cfg.field.use_finite_spectral}")
    if "finite_spectral_modes" in net_cfg_yaml:
        cfg.model_cfg.field.finite_spectral_modes = int(net_cfg_yaml["finite_spectral_modes"])
        print(f"[main] Finite spectral modes: {cfg.model_cfg.field.finite_spectral_modes}")
    if "finite_spectral_with_distance" in net_cfg_yaml:
        cfg.model_cfg.field.finite_spectral_with_distance = bool(net_cfg_yaml["finite_spectral_with_distance"])
    if "use_engineering_semantics" in net_cfg_yaml:
        cfg.model_cfg.field.use_engineering_semantics = bool(net_cfg_yaml["use_engineering_semantics"])
        print(f"[main] Engineering semantics: {cfg.model_cfg.field.use_engineering_semantics}")
    if "semantic_feat_dim" in net_cfg_yaml:
        cfg.model_cfg.field.semantic_feat_dim = int(net_cfg_yaml["semantic_feat_dim"])
        print(f"[main] Semantic feature dim: {cfg.model_cfg.field.semantic_feat_dim}")
    if "strict_mixed_default_eps_bridge" in net_cfg_yaml:
        cfg.model_cfg.field.strict_mixed_default_eps_bridge = bool(net_cfg_yaml["strict_mixed_default_eps_bridge"])
        print(f"[main] Strict mixed default eps bridge: {cfg.model_cfg.field.strict_mixed_default_eps_bridge}")
    if "strict_mixed_contact_pointwise_stress" in net_cfg_yaml:
        cfg.model_cfg.field.strict_mixed_contact_pointwise_stress = bool(
            net_cfg_yaml["strict_mixed_contact_pointwise_stress"]
        )
        print(
            "[main] Strict mixed contact pointwise stress: "
            f"{cfg.model_cfg.field.strict_mixed_contact_pointwise_stress}"
        )
    if "inner_contact_state_adapter_enabled" in net_cfg_yaml:
        cfg.model_cfg.field.inner_contact_state_adapter_enabled = bool(
            net_cfg_yaml["inner_contact_state_adapter_enabled"]
        )
        print(
            "[main] Inner contact state adapter: "
            f"{cfg.model_cfg.field.inner_contact_state_adapter_enabled}"
        )
    if "internal_ring_lift_enabled" in net_cfg_yaml:
        cfg.model_cfg.field.internal_ring_lift_enabled = bool(net_cfg_yaml["internal_ring_lift_enabled"])
    if "internal_ring_center" in net_cfg_yaml:
        raw = net_cfg_yaml.get("internal_ring_center") or ()
        if len(raw) != 2:
            raise ValueError("network_config.internal_ring_center must contain exactly two values.")
        cfg.model_cfg.field.internal_ring_center = (float(raw[0]), float(raw[1]))
    if "internal_ring_r_in" in net_cfg_yaml:
        cfg.model_cfg.field.internal_ring_r_in = float(net_cfg_yaml["internal_ring_r_in"])
    if "internal_ring_r_out" in net_cfg_yaml:
        cfg.model_cfg.field.internal_ring_r_out = float(net_cfg_yaml["internal_ring_r_out"])
    if "internal_ring_fourier_order" in net_cfg_yaml:
        cfg.model_cfg.field.internal_ring_fourier_order = int(net_cfg_yaml["internal_ring_fourier_order"])
    if "cylindrical_primary_head_enabled" in net_cfg_yaml:
        cfg.model_cfg.field.cylindrical_primary_head_enabled = bool(net_cfg_yaml["cylindrical_primary_head_enabled"])
    if "annular_modal_residual_enabled" in net_cfg_yaml:
        cfg.model_cfg.field.annular_modal_residual_enabled = bool(
            net_cfg_yaml["annular_modal_residual_enabled"]
        )
    if "annular_modal_residual_center" in net_cfg_yaml:
        raw = net_cfg_yaml.get("annular_modal_residual_center") or ()
        if len(raw) != 2:
            raise ValueError("network_config.annular_modal_residual_center must contain exactly two values.")
        cfg.model_cfg.field.annular_modal_residual_center = (float(raw[0]), float(raw[1]))
    if "annular_modal_residual_r_in" in net_cfg_yaml:
        cfg.model_cfg.field.annular_modal_residual_r_in = float(
            net_cfg_yaml["annular_modal_residual_r_in"]
        )
    if "annular_modal_residual_r_out" in net_cfg_yaml:
        cfg.model_cfg.field.annular_modal_residual_r_out = float(
            net_cfg_yaml["annular_modal_residual_r_out"]
        )
    if "annular_modal_residual_radial_order" in net_cfg_yaml:
        cfg.model_cfg.field.annular_modal_residual_radial_order = int(
            net_cfg_yaml["annular_modal_residual_radial_order"]
        )
    if "annular_modal_residual_fourier_order" in net_cfg_yaml:
        cfg.model_cfg.field.annular_modal_residual_fourier_order = int(
            net_cfg_yaml["annular_modal_residual_fourier_order"]
        )
    if "annular_modal_residual_max_amplitude" in net_cfg_yaml:
        cfg.model_cfg.field.annular_modal_residual_max_amplitude = float(
            net_cfg_yaml["annular_modal_residual_max_amplitude"]
        )
    if "annular_modal_residual_target_component" in net_cfg_yaml:
        cfg.model_cfg.field.annular_modal_residual_target_component = int(
            net_cfg_yaml["annular_modal_residual_target_component"]
        )
    if bool(getattr(cfg.model_cfg.field, "annular_modal_residual_enabled", False)):
        print(
            "[main] Annular modal residual: "
            f"r=[{cfg.model_cfg.field.annular_modal_residual_r_in:g}, "
            f"{cfg.model_cfg.field.annular_modal_residual_r_out:g}], "
            f"radial_order={cfg.model_cfg.field.annular_modal_residual_radial_order}, "
            f"fourier_order={cfg.model_cfg.field.annular_modal_residual_fourier_order}, "
            f"max_amp={cfg.model_cfg.field.annular_modal_residual_max_amplitude:g}, "
            f"component={cfg.model_cfg.field.annular_modal_residual_target_component}"
        )
    if (
        bool(getattr(cfg.model_cfg.field, "use_engineering_semantics", False))
        and int(getattr(cfg.model_cfg.field, "semantic_feat_dim", 0) or 0) <= 0
    ):
        cfg.model_cfg.field.semantic_feat_dim = 4
        print("[main] Semantic feature dim defaulted to 4 (contact/bc/mirror/material).")
    if "uncertainty_out_dim" in net_cfg_yaml:
        cfg.model_cfg.field.uncertainty_out_dim = int(net_cfg_yaml["uncertainty_out_dim"])
        print(f"[main] Uncertainty head out dim: {cfg.model_cfg.field.uncertainty_out_dim}")
    if "use_graph" in net_cfg_yaml:
        cfg.model_cfg.field.use_graph = bool(net_cfg_yaml["use_graph"])
    if "adaptive_depth_enabled" in net_cfg_yaml:
        cfg.model_cfg.field.adaptive_depth_enabled = bool(net_cfg_yaml["adaptive_depth_enabled"])
    if "adaptive_depth_mode" in net_cfg_yaml:
        cfg.model_cfg.field.adaptive_depth_mode = str(net_cfg_yaml["adaptive_depth_mode"])
    if "adaptive_depth_shallow_layers" in net_cfg_yaml:
        cfg.model_cfg.field.adaptive_depth_shallow_layers = int(net_cfg_yaml["adaptive_depth_shallow_layers"])
    if "adaptive_depth_threshold" in net_cfg_yaml:
        cfg.model_cfg.field.adaptive_depth_threshold = float(net_cfg_yaml["adaptive_depth_threshold"])
    if "adaptive_depth_temperature" in net_cfg_yaml:
        cfg.model_cfg.field.adaptive_depth_temperature = float(net_cfg_yaml["adaptive_depth_temperature"])
    if "adaptive_depth_route_source" in net_cfg_yaml:
        cfg.model_cfg.field.adaptive_depth_route_source = str(net_cfg_yaml["adaptive_depth_route_source"])

    # ===== 接触力学参数（normal/friction）=====
    normal_cfg_yaml = cfg_yaml.get("normal_config", {}) or {}
    if isinstance(normal_cfg_yaml, dict) and normal_cfg_yaml:
        if "beta" in normal_cfg_yaml:
            cfg.contact_cfg.normal.beta = float(normal_cfg_yaml["beta"])
        if "mu_n" in normal_cfg_yaml:
            cfg.contact_cfg.normal.mu_n = float(normal_cfg_yaml["mu_n"])
        if "mode" in normal_cfg_yaml:
            cfg.contact_cfg.normal.mode = str(normal_cfg_yaml["mode"])
        if "residual_mode" in normal_cfg_yaml:
            cfg.contact_cfg.normal.residual_mode = str(normal_cfg_yaml["residual_mode"])
        if "fb_eps" in normal_cfg_yaml:
            cfg.contact_cfg.normal.fb_eps = float(normal_cfg_yaml["fb_eps"])

    fric_cfg_yaml = cfg_yaml.get("friction_config", {}) or {}
    if isinstance(fric_cfg_yaml, dict) and fric_cfg_yaml:
        if "enabled" in fric_cfg_yaml:
            cfg.contact_cfg.friction.enabled = bool(fric_cfg_yaml["enabled"])
        if "k_t" in fric_cfg_yaml:
            cfg.contact_cfg.friction.k_t = float(fric_cfg_yaml["k_t"])
        if "mu_t" in fric_cfg_yaml:
            cfg.contact_cfg.friction.mu_t = float(fric_cfg_yaml["mu_t"])
        if "mu_f" in fric_cfg_yaml:
            cfg.contact_cfg.friction.mu_f = float(fric_cfg_yaml["mu_f"])
        if "use_bipotential_residual" in fric_cfg_yaml:
            cfg.contact_cfg.friction.use_bipotential_residual = bool(fric_cfg_yaml["use_bipotential_residual"])
        if "bipotential_weight" in fric_cfg_yaml:
            cfg.contact_cfg.friction.bipotential_weight = float(fric_cfg_yaml["bipotential_weight"])
        if "bipotential_eps" in fric_cfg_yaml:
            cfg.contact_cfg.friction.bipotential_eps = float(fric_cfg_yaml["bipotential_eps"])
        if "use_smooth_friction" in fric_cfg_yaml:
            val = bool(fric_cfg_yaml["use_smooth_friction"])
            cfg.contact_cfg.use_smooth_friction = val
            cfg.contact_cfg.friction.use_smooth_friction = val
        if "use_delta_st_friction" in fric_cfg_yaml:
            cfg.contact_cfg.friction.use_delta_st = bool(fric_cfg_yaml["use_delta_st_friction"])

    adaptive_cfg = loss_cfg_yaml.get("adaptive", {}) or {}
    cfg.loss_adaptive_enabled = bool(
        adaptive_cfg.get("enabled", cfg.loss_adaptive_enabled)
    )
    cfg.loss_update_every = int(adaptive_cfg.get("update_every", cfg.loss_update_every))
    cfg.loss_ema_decay = float(adaptive_cfg.get("ema_decay", cfg.loss_ema_decay))
    # 绝对权重上下限（建议用该方式约束自适应权重，避免出现危险的超大权重）
    if "min_weight" in adaptive_cfg:
        cfg.loss_min_weight = float(adaptive_cfg["min_weight"])
    if "max_weight" in adaptive_cfg:
        cfg.loss_max_weight = float(adaptive_cfg["max_weight"])
    # 每次更新时的相对缩放因子上下限（可选；默认 0.25~4.0）
    if "min_factor" in adaptive_cfg:
        cfg.loss_min_factor = float(adaptive_cfg["min_factor"])
    if "max_factor" in adaptive_cfg:
        cfg.loss_max_factor = float(adaptive_cfg["max_factor"])
    temperature = float(adaptive_cfg.get("temperature", 0.0) or 0.0)
    if temperature > 0.0:
        cfg.loss_gamma = 1.0 / temperature
    else:
        cfg.loss_gamma = float(adaptive_cfg.get("gamma", cfg.loss_gamma))

    focus_terms_yaml = adaptive_cfg.get("focus_terms", []) or []
    focus_terms = []
    skip_supervision_focus = bool(getattr(cfg.supervision, "enabled", False))
    skipped_focus_terms = []
    for item in focus_terms_yaml:
        key = str(item).strip()
        mapping = weight_key_map.get(key)
        if mapping is None:
            continue
        loss_key = mapping[1]
        if skip_supervision_focus and loss_key == "E_data":
            skipped_focus_terms.append(key)
            continue
        focus_terms.append(loss_key)
    cfg.loss_focus_terms = tuple(focus_terms)
    if skipped_focus_terms:
        skipped = ", ".join(skipped_focus_terms)
        print(f"[main] Adaptive focus skips supervision term(s): {skipped} -> E_data")
    cfg.total_cfg.adaptive_scheme = adaptive_cfg.get("scheme", cfg.total_cfg.adaptive_scheme)

    ed_cfg = loss_cfg_yaml.get("energy_dissipation", {}) or {}
    if isinstance(ed_cfg, dict) and ed_cfg:
        if "enabled" in ed_cfg:
            cfg.total_cfg.ed_enabled = bool(ed_cfg["enabled"])
        if "external_scale" in ed_cfg:
            cfg.total_cfg.ed_external_scale = float(ed_cfg["external_scale"])
        if "margin" in ed_cfg:
            cfg.total_cfg.ed_margin = float(ed_cfg["margin"])
        if "use_relu" in ed_cfg:
            cfg.total_cfg.ed_use_relu = bool(ed_cfg["use_relu"])
        if "squared" in ed_cfg:
            cfg.total_cfg.ed_square = bool(ed_cfg["squared"])
    if float(getattr(cfg.total_cfg, "w_ed", 0.0) or 0.0) > 0.0 and "enabled" not in ed_cfg:
        cfg.total_cfg.ed_enabled = True

    if "supervision_contribution_floor_enabled" in loss_cfg_yaml:
        cfg.supervision_contribution_floor_enabled = bool(
            loss_cfg_yaml["supervision_contribution_floor_enabled"]
        )
    if "supervision_contribution_floor_ratio" in loss_cfg_yaml:
        cfg.supervision_contribution_floor_ratio = float(
            loss_cfg_yaml["supervision_contribution_floor_ratio"]
        )
    if "data_smoothing_k" in loss_cfg_yaml:
        cfg.total_cfg.data_smoothing_k = int(loss_cfg_yaml["data_smoothing_k"])
    if "data_weight_enabled" in loss_cfg_yaml:
        cfg.total_cfg.data_weight_enabled = bool(loss_cfg_yaml["data_weight_enabled"])
        print(f"[main] Morphology-weighted supervision: {cfg.total_cfg.data_weight_enabled}")
    if "data_weight_blend" in loss_cfg_yaml:
        cfg.total_cfg.data_weight_blend = float(loss_cfg_yaml["data_weight_blend"])
    if "data_weight_power" in loss_cfg_yaml:
        cfg.total_cfg.data_weight_power = float(loss_cfg_yaml["data_weight_power"])
    optical_modal_yaml = loss_cfg_yaml.get("optical_modal", {}) or {}
    if isinstance(optical_modal_yaml, dict) and optical_modal_yaml:
        if "enabled" in optical_modal_yaml:
            cfg.total_cfg.optical_modal_enabled = bool(optical_modal_yaml["enabled"])
        if "center" in optical_modal_yaml and optical_modal_yaml["center"] is not None:
            center = list(optical_modal_yaml["center"])
            if len(center) >= 2:
                cfg.total_cfg.optical_modal_center = (float(center[0]), float(center[1]))
        if "r_in" in optical_modal_yaml:
            cfg.total_cfg.optical_modal_r_in = float(optical_modal_yaml["r_in"])
        if "r_out" in optical_modal_yaml:
            cfg.total_cfg.optical_modal_r_out = float(optical_modal_yaml["r_out"])
        if "radial_order" in optical_modal_yaml:
            cfg.total_cfg.optical_modal_radial_order = int(optical_modal_yaml["radial_order"])
        if "fourier_order" in optical_modal_yaml:
            cfg.total_cfg.optical_modal_fourier_order = int(optical_modal_yaml["fourier_order"])
        if "target_component" in optical_modal_yaml:
            cfg.total_cfg.optical_modal_target_component = int(optical_modal_yaml["target_component"])

    unc_cfg = cfg_yaml.get("uncertainty_config", {}) or {}
    if isinstance(unc_cfg, dict) and unc_cfg:
        if "loss_weight" in unc_cfg:
            cfg.uncertainty_loss_weight = float(unc_cfg["loss_weight"])
        if "sample_points" in unc_cfg:
            cfg.uncertainty_sample_points = int(unc_cfg["sample_points"])
        if "proxy_scale" in unc_cfg:
            cfg.uncertainty_proxy_scale = float(unc_cfg["proxy_scale"])
        if "logvar_min" in unc_cfg:
            cfg.uncertainty_logvar_min = float(unc_cfg["logvar_min"])
        if "logvar_max" in unc_cfg:
            cfg.uncertainty_logvar_max = float(unc_cfg["logvar_max"])

    # allow loss_config.base_weights.w_unc as shorthand
    if "w_unc" in base_weights_yaml and float(cfg.uncertainty_loss_weight) <= 0.0:
        cfg.uncertainty_loss_weight = float(base_weights_yaml["w_unc"])

    # 启用应力头时默认也纳入自适应关注项，避免固定权重过大导致梯度爆炸
    has_stress_head = getattr(cfg.model_cfg.field, "stress_out_dim", 0) > 0
    if has_stress_head and "E_sigma" not in cfg.loss_focus_terms:
        cfg.loss_focus_terms = tuple(list(cfg.loss_focus_terms) + ["E_sigma"])

    # 只要存在任意关注项，就默认使用“平衡”策略（也可在 config.yaml 中通过 adaptive.scheme 显式指定）
    if cfg.loss_focus_terms:
        scheme_norm = str(getattr(cfg.total_cfg, "adaptive_scheme", "") or "").strip().lower()
        if scheme_norm in {"", "contact_only", "basic"}:
            cfg.total_cfg.adaptive_scheme = "balance"
    cfg.alm_update_every = int(cfg_yaml.get("alm_update_every", cfg.alm_update_every))

    if cfg.incremental_mode:
        cfg.contact_cfg.update_every_steps = 1
        cfg.alm_update_every = 0


    # ===== 显存友好覆盖（建议先这样跑通，再逐步调回） =====
    debug_big_model = bool(cfg_yaml.get("debug_big_model", False))
    if debug_big_model:
        # 1) 提升模型表达能力（更宽更深的位移网络 + 更大的条件编码器）
        cfg.model_cfg.encoder.width = 96
        cfg.model_cfg.encoder.depth = 3
        cfg.model_cfg.encoder.out_dim = 96
        cfg.model_cfg.field.width = 320
        cfg.model_cfg.field.depth = 9
        cfg.model_cfg.field.residual_skips = (3, 6, 8)

    # 2) DFEM 采样配置（不再设置 Jacobian 相关字段）
    #    - chunk_size: 节点前向/能量评估的分块大小（防止一次性吃满显存）
    #    - n_points_per_step: 每一步参与 DFEM 积分的子单元/积分点个数上限
    cfg.elas_cfg.chunk_size = int(elas_cfg_yaml.get("chunk_size", 0))
    raw_n_points = elas_cfg_yaml.get("n_points_per_step", 4096)
    if raw_n_points is None:
        cfg.elas_cfg.n_points_per_step = None
    else:
        cfg.elas_cfg.n_points_per_step = int(raw_n_points)
    cfg.elas_cfg.coord_scale = float(elas_cfg_yaml.get("coord_scale", 1.0))
    if "stress_loss_weight" in elas_cfg_yaml:
        cfg.elas_cfg.stress_loss_weight = float(elas_cfg_yaml.get("stress_loss_weight", cfg.elas_cfg.stress_loss_weight))
    if "use_forward_mode" in elas_cfg_yaml:
        cfg.elas_cfg.use_forward_mode = bool(elas_cfg_yaml.get("use_forward_mode"))
    if "cache_sample_metrics" in elas_cfg_yaml:
        cfg.elas_cfg.cache_sample_metrics = bool(elas_cfg_yaml.get("cache_sample_metrics"))
    else:
        cfg.elas_cfg.cache_sample_metrics = False

    # Canonicalize route-dependent switches before any stage-dependent heuristics.
    if cfg.training_profile == _LOCKED_TRAINING_PROFILE:
        _canonicalize_locked_route(cfg)
    elif cfg.training_profile == _NORMAL_CONTACT_FIRST_MAINLINE_PROFILE:
        _canonicalize_normal_contact_first_mainline(cfg)
    elif cfg.training_profile == _P3_LEARNING_GATE_PROFILE:
        _canonicalize_p3_learning_gate(cfg)
    elif cfg.training_profile == _P5A_REENTRY_GATE_PROFILE:
        _canonicalize_p5a_reentry_gate(cfg)
    elif cfg.training_profile == _P5B1_PHYSICS_REENTRY_GATE_PROFILE:
        _canonicalize_p5b1_physics_reentry_gate(cfg)

    # 3) 接触/拧紧采样：根据阶段数做显存友好的调整
    stage_multiplier = 1
    if cfg.preload_use_stages:
        stage_multiplier = max(1, len(cfg.preload_specs))
        if cfg.preload_sequence:
            for entry in cfg.preload_sequence:
                if isinstance(entry, dict):
                    order = entry.get("order") or entry.get("orders")
                    values = entry.get("values") or entry.get("P")
                    if order is not None:
                        stage_multiplier = max(stage_multiplier, len(order))
                    elif values is not None:
                        stage_multiplier = max(stage_multiplier, len(values))
                elif isinstance(entry, (list, tuple)):
                    stage_multiplier = max(stage_multiplier, len(entry))

    # 载荷跨度（用于适当放大/缩小每阶段采样规模）
    load_span = float(abs(getattr(cfg, "preload_max", 0.0) - getattr(cfg, "preload_min", 0.0)))
    unit = str(getattr(cfg.tightening_cfg, "angle_unit", "deg") or "deg").lower()
    ref_span = 30.0 if unit.startswith("deg") else 0.5  # ??????? 30deg / 0.5rad?
    load_factor = 1.0
    if ref_span > 0:
        load_factor = min(2.0, max(0.5, load_span / ref_span))
    if stage_multiplier > 1 and abs(load_factor - 1.0) > 1e-3:
        print(f"[main] 拧紧角度跨度 {load_span:g} -> 每阶段采样系数 {load_factor:.2f}")

    # 分阶段加载时，ContactOperator 内部的 update_every_steps 会被每阶段多次调用，
    # 这里按阶段数放大一次频率，保持与单阶段训练相近的物理更新节奏。
    if stage_multiplier > 1 and not cfg.incremental_mode:
        try:
            cfg.contact_cfg.update_every_steps = int(
                max(1, cfg.contact_cfg.update_every_steps * stage_multiplier)
            )
            cfg.alm_update_every = int(
                max(1, cfg.alm_update_every * stage_multiplier)
            )
            print(
                f"[main] 分阶段拧紧启用：ALM 更新频率放宽为每 {cfg.alm_update_every} 步一次，"
                f"ContactOperator.update_every_steps={cfg.contact_cfg.update_every_steps}"
            )
        except Exception:
            pass

    contact_target = cfg.n_contact_points_per_pair
    if contact_mode_norm == "mortar" and contact_mortar_max_points > 0:
        contact_target = cfg.contact_mortar_max_points

    if stage_multiplier > 1 and not (contact_mode_norm == "mortar" and contact_mortar_max_points <= 0):
        per_stage_contact = max(256, math.ceil(contact_target / stage_multiplier))
        per_stage_contact = max(256, int(math.ceil(per_stage_contact * load_factor)))
        approx_total_contact = per_stage_contact * stage_multiplier
        if per_stage_contact != contact_target:
            if contact_mode_norm == "mortar" and contact_mortar_max_points > 0:
                print(
                    "[main] 分阶段拧紧启用：将 mortar 接触上限从 "
                    f"{contact_target} 调整为每阶段 {per_stage_contact} (≈{approx_total_contact} 总点数)。"
                )
            else:
                print(
                    "[main] 分阶段拧紧启用：将每对接触采样从 "
                    f"{contact_target} 调整为每阶段 {per_stage_contact} (≈{approx_total_contact} 总点数)。"
                )
        # 分阶段计算仍会在同一梯度带内重复评估接触能，因此进一步限制总量
        contact_cap = 2048
        if per_stage_contact > contact_cap:
            per_stage_contact = contact_cap
            approx_total_contact = per_stage_contact * stage_multiplier
            print(
                "[main] 接触点上限触发：将每阶段采样压缩到 "
                f"{per_stage_contact} (≈{approx_total_contact} 总点数)。"
            )
        if contact_mode_norm == "mortar" and contact_mortar_max_points > 0:
            cfg.contact_mortar_max_points = per_stage_contact
        else:
            cfg.n_contact_points_per_pair = per_stage_contact

        preload_target = cfg.preload_n_points_each
        per_stage_preload = max(128, math.ceil(preload_target / stage_multiplier))
        per_stage_preload = max(128, int(math.ceil(per_stage_preload * load_factor)))
        approx_total_preload = per_stage_preload * stage_multiplier
        if per_stage_preload != preload_target:
            print(
                "[main] 分阶段拧紧启用：将每个螺母端面的采样从 "
                f"{preload_target} 调整为每阶段 {per_stage_preload} (≈{approx_total_preload} 总点数)。"
            )
        preload_cap = 1024
        if per_stage_preload > preload_cap:
            per_stage_preload = preload_cap
            approx_total_preload = per_stage_preload * stage_multiplier
            print(
                "[main] 拧紧点上限触发：将每阶段端面采样压缩到 "
                f"{per_stage_preload} (≈{approx_total_preload} 总点数)。"
            )
        cfg.preload_n_points_each = per_stage_preload

        elas_target = cfg.elas_cfg.n_points_per_step
        if elas_target is not None:
            try:
                if float(elas_target) <= 0:
                    elas_target = None
            except Exception:
                pass
        if elas_target is not None:
            per_stage_elas = max(1024, math.ceil(float(elas_target) / stage_multiplier))
            per_stage_elas = max(1024, int(math.ceil(per_stage_elas * load_factor)))
            if per_stage_elas != elas_target:
                print(
                    "[main] 分阶段拧紧启用：将 DFEM 每步积分点从 "
                    f"{elas_target} 调整为每阶段 {per_stage_elas}。"
                )
                cfg.elas_cfg.n_points_per_step = per_stage_elas


    # 5) 根据拧紧角度范围自动调整归一化（映射到约 [-1, 1]）
    preload_lo, preload_hi = float(cfg.preload_min), float(cfg.preload_max)
    if preload_hi < preload_lo:
        raise ValueError("拧紧角度范围 tighten_angle_range 的上限必须大于下限。")
    preload_mid = 0.5 * (preload_lo + preload_hi)
    preload_half_span = 0.5 * (preload_hi - preload_lo)
    cfg.model_cfg.preload_shift = preload_mid
    cfg.model_cfg.preload_scale = max(preload_half_span, 1e-3)
    if cfg.training_profile == _LOCKED_TRAINING_PROFILE:
        _enforce_locked_route(cfg)
        print("[main] Locked route:", _LOCKED_ROUTE_NAME)
    elif cfg.training_profile == _NORMAL_CONTACT_FIRST_MAINLINE_PROFILE:
        _validate_normal_contact_first_mainline_profile(cfg)
        print("[main] Normal-contact-first mainline route:", cfg.training_profile)
        print(
            "[main] Coupling-tightening protocol:",
            getattr(cfg, "coupling_tightening_protocol", _NORMAL_CONTACT_TIGHTENING_PROTOCOL),
        )
    elif cfg.training_profile in {_STRICT_MIXED_EXPERIMENTAL_PROFILE, _STRICT_MIXED_POST_REENTRY_PROFILE}:
        _validate_experimental_profile(cfg)
        label = "Post-reentry route" if cfg.training_profile == _STRICT_MIXED_POST_REENTRY_PROFILE else "Experimental route"
        print(f"[main] {label}:", cfg.training_profile)
    elif cfg.training_profile == _P3_LEARNING_GATE_PROFILE:
        _validate_p3_learning_gate_profile(cfg)
        print("[main] P3 route:", cfg.training_profile)
    elif cfg.training_profile == _P5A_REENTRY_GATE_PROFILE:
        _validate_p5a_reentry_gate_profile(cfg)
        print("[main] P5a route:", cfg.training_profile)
    else:
        _validate_p5b1_physics_reentry_gate_profile(cfg)
        print("[main] P5b1 route:", cfg.training_profile)
    # =================================================
    return cfg, asm


def _run_training(cfg, asm, export_saved_model: str = ""):
    from train.trainer import Trainer  # 再导一次确保路径就绪

    # 为本次训练创建带时间戳的独立 checkpoint 目录，避免文件占用冲突
    base_ckpt_dir = cfg.ckpt_dir or "checkpoints"
    ts_tag = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    candidate = os.path.join(base_ckpt_dir, ts_tag)
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(base_ckpt_dir, f"{ts_tag}-{suffix}")
        suffix += 1
    os.makedirs(candidate, exist_ok=True)
    cfg.ckpt_dir = candidate
    print(f"[main] 本次训练的 checkpoint 输出目录：{cfg.ckpt_dir}")

    trainer = Trainer(cfg)
    trainer.run()

    export_dir = (export_saved_model or "").strip()
    if export_dir:
        export_dir = os.path.abspath(export_dir)
        os.makedirs(os.path.dirname(export_dir), exist_ok=True)
    else:
        export_dir = _default_saved_model_dir(cfg.out_dir)
        print(f"[main] 未提供 --export，将 SavedModel 写入: {export_dir}")
    trainer.export_saved_model(export_dir)

    out_dir_disp = cfg.out_dir or "outputs"
    print(
        f"\n[OK] 训练完成！请到 '{out_dir_disp}' 查看镜面变形云图（数量取决于可视化配置）。"
    )
    print("   如需修改 CDB 路径、表面名或超参，请编辑 config.yaml。")
def _run_single_training_phase(cfg, asm, export_saved_model: str = ""):
    from train.trainer import Trainer  # 鍐嶅涓€娆＄‘淇濊矾寰勫氨缁?

    phase_name = str(getattr(cfg, "run_phase_name", "") or "single")
    cfg.ckpt_dir = _allocate_run_checkpoint_dir(cfg.ckpt_dir)
    print(f"[main] {phase_name}: checkpoint dir = {cfg.ckpt_dir}")

    trainer = Trainer(cfg)
    trainer.run()

    export_dir = _resolve_export_dir(cfg, export_saved_model)
    if export_dir:
        trainer.export_saved_model(export_dir)
        print(f"[main] {phase_name}: SavedModel exported to {export_dir}")
    else:
        print(f"[main] {phase_name}: SavedModel export skipped")

    result = _TrainingPhaseResult(
        phase_name=phase_name,
        trainer=trainer,
        best_ckpt_path=str(getattr(trainer, "_best_ckpt_path", "") or ""),
        final_ckpt_path=str(getattr(trainer, "_final_ckpt_path", "") or ""),
        out_dir=str(getattr(cfg, "out_dir", "") or ""),
        ckpt_dir=str(getattr(cfg, "ckpt_dir", "") or ""),
        export_dir=str(export_dir or ""),
    )
    summary_path = _resolve_paper_benchmark_summary_path(cfg)
    benchmark_meta = getattr(cfg, "benchmark_meta", None)
    if summary_path:
        summary = build_paper_benchmark_summary(
            benchmark_meta if benchmark_meta is not None else cfg,
            result,
        )
        written_summary_path = write_paper_benchmark_summary(summary, summary_path)
        result.summary_path = str(written_summary_path)
        print(f"[main] {phase_name}: benchmark summary written to {written_summary_path}")
    print(f"[main] {phase_name}: completed; best={result.best_ckpt_path or '-'} final={result.final_ckpt_path or '-'}")
    return result


def _run_two_stage_training(cfg, asm, export_saved_model: str = ""):
    print("[main] Two-stage training enabled: phase1 -> phase2")
    phase1_cfg = _derive_phase_config(cfg, "phase1")
    phase1_result = _run_single_training_phase(phase1_cfg, asm, export_saved_model=None)

    phase1_best = str(getattr(phase1_result, "best_ckpt_path", "") or "").strip()
    if not phase1_best:
        raise RuntimeError("Phase 1 did not produce a best checkpoint for Phase 2 handoff.")

    print(f"[main] Phase 1 handoff checkpoint: {phase1_best}")
    phase2_cfg = _derive_phase_config(cfg, "phase2")
    phase2_cfg.resume_ckpt_path = phase1_best
    print(f"[main] Phase 2 resume checkpoint: {phase2_cfg.resume_ckpt_path}")
    phase2_result = _run_single_training_phase(phase2_cfg, asm, export_saved_model=export_saved_model)

    out_dir_disp = str(getattr(phase2_result, "out_dir", "") or "outputs")
    print(f"\n[OK] Two-stage training finished. Outputs: '{out_dir_disp}'")
    print("   Edit config.yaml to change the CDB path, surface names, or hyperparameters.")
    return {"phase1": phase1_result, "phase2": phase2_result}


def _run_training(cfg, asm, export_saved_model: str = ""):
    if bool(getattr(getattr(cfg, "two_stage_training", None), "enabled", False)):
        return _run_two_stage_training(cfg, asm, export_saved_model=export_saved_model)

    result = _run_single_training_phase(cfg, asm, export_saved_model=export_saved_model)
    out_dir_disp = str(getattr(result, "out_dir", "") or "outputs")
    print(f"\n[OK] Training finished. Outputs: '{out_dir_disp}'")
    print("   Edit config.yaml to change the CDB path, surface names, or hyperparameters.")
    return result


def main(argv=None):
    _setup_run_logs()
    parser = argparse.ArgumentParser(
        description="Train the DFEM/PINN model."
    )
    parser.add_argument(
        "--config", default="",
        help="可选配置文件路径；默认读取仓库根目录下的 config.yaml。"
    )
    parser.add_argument(
        "--export", default="",
        help="将模型导出为 TensorFlow SavedModel 的目录"
    )

    args = parser.parse_args(argv)

    cfg, asm = _prepare_config_with_autoguess(config_path=args.config or None)
    _run_training(cfg, asm, export_saved_model=args.export)


if __name__ == "__main__":
    main()
