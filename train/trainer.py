# -*- coding: utf-8 -*-
"""
trainer.py 鈥?涓昏缁冨惊鐜紙绮剧畝鏃ュ織 + 鍒嗛樁娈佃繘搴︽彁绀猴級銆?

璇ョ増鏈笓娉ㄤ簬淇濈暀鍏抽敭鏋勫缓/璁粌淇℃伅锛?
  - 鍒濆鍖栨椂鎶ュ憡鏄惁鍚敤 GPU銆?
  - 鏋勫缓闃舵浠呰緭鍑哄繀闇€鐨勪俊鎭笌鎺ヨЕ姹囨€汇€?
  - 鍗曟璁粌杩涘害鏉′細鏍囨敞褰撳墠闃舵锛屼究浜庤瀵熻缁冩祦绋嬨€?
"""
from __future__ import annotations

import os
import sys
import time
import re
import math
import random
import shutil
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

# ---------- TF 鏄惧瓨涓庡垎閰嶅櫒 ----------
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

try:
    import colorama
    colorama.just_fix_windows_console()
    _ANSI_WHITE = colorama.Fore.WHITE
    _ANSI_RESET = colorama.Style.RESET_ALL
except Exception:
    colorama = None
    _ANSI_WHITE = ""
    _ANSI_RESET = ""

import builtins as _builtins


def _wrap_white(text: str) -> str:
    if not _ANSI_WHITE:
        return text
    return f"{_ANSI_WHITE}{text}{_ANSI_RESET}"


def print(*values, sep: str = " ", end: str = "\n", file=None, flush: bool = False):
    """Module-local print that forces white foreground text on stdout/stderr."""

    target = sys.stdout if file is None else file
    msg = sep.join(str(v) for v in values)
    if target in (sys.stdout, sys.stderr):
        msg = _wrap_white(msg)
    try:
        _builtins.print(msg, end=end, file=target, flush=flush)
    except UnicodeEncodeError:
        # Windows consoles may use legacy encodings (e.g. GBK) and crash on
        # special Unicode symbols. Keep Chinese text if possible, escape only
        # the unencodable characters.
        enc = getattr(target, "encoding", None) or "utf-8"
        safe = msg.encode(enc, errors="backslashreplace").decode(enc, errors="ignore")
        _builtins.print(safe, end=end, file=target, flush=flush)

# 璁?src 鏍圭洰褰曞彲瀵煎叆
_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

# ---------- 椤圭洰妯″潡 ----------
from inp_io.inp_parser import AssemblyModel
from mesh.contact_pairs import ContactPairSpec
from physics.material_lib import MaterialLibrary
from physics.elasticity_residual import ElasticityResidual
from physics.contact.contact_operator import ContactOperator
from physics.boundary_conditions import BoundaryPenalty
from physics.tightening_model import NutTighteningPenalty
from model.loss_energy import TotalEnergy
from train.loss_weights import LossWeightState
from train.trainer_config import TrainerConfig
from train.trainer_build_mixin import TrainerBuildMixin
from train.trainer_preload_mixin import TrainerPreloadMixin
from train.trainer_init_mixin import TrainerInitMixin
from train.trainer_monitor_mixin import TrainerMonitorMixin
from train.trainer_viz_mixin import TrainerVizMixin
from train.trainer_run_mixin import TrainerRunMixin
from train.trainer_opt_mixin import TrainerOptMixin
from train import normal_contact_training_protocol
from train.saved_model_module import _SavedModelModule


def _find_node_id_in_boundary_raw(raw: str) -> Optional[int]:
    """Extract node id from boundary raw text (supports CDB/Abaqus-like formats)."""

    txt = str(raw or "").strip()
    if not txt:
        return None
    m = re.match(r"^\s*D\s*,\s*([+-]?\d+)", txt, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    for tok in re.split(r"[,\s]+", txt):
        if not tok:
            continue
        try:
            return int(float(tok))
        except Exception:
            continue
    return None


def _find_boundary_dof_flags(raw: str) -> Tuple[float, float, float]:
    """Extract boundary DOF flags for UX/UY/UZ-like constraints."""

    txt = str(raw or "").strip().upper()
    if not txt:
        return 0.0, 0.0, 0.0
    tokens = [tok.strip().upper() for tok in re.split(r"[,\s]+", txt) if tok.strip()]
    if not tokens:
        return 0.0, 0.0, 0.0

    dof_token = tokens[2] if len(tokens) >= 3 and tokens[0] == "D" else ""
    if dof_token in {"ALL", "U", "UXYZ", "XYZ"}:
        return 1.0, 1.0, 1.0

    ux = float("UX" in tokens or dof_token == "X")
    uy = float("UY" in tokens or dof_token == "Y")
    uz = float("UZ" in tokens or dof_token == "Z")
    return ux, uy, uz


def build_node_semantic_features(
    asm: AssemblyModel,
    sorted_node_ids: np.ndarray,
    part2mat: Mapping[str, str],
    mirror_surface_name: str = "MIRROR UP",
) -> np.ndarray:
    """Build CDB engineering semantic features aligned with sorted global node ids.

    Feature layout (N, 8):
    - [:,0] contact flag
    - [:,1] boundary-condition flag
    - [:,2] mirror-region flag
    - [:,3] normalized material id (0~1)
    - [:,4] UX-constrained flag
    - [:,5] UY-constrained flag
    - [:,6] UZ-constrained flag
    - [:,7] generic surface-like flag (contact or constrained or mirror)
    """

    node_ids = np.asarray(sorted_node_ids, dtype=np.int64).reshape(-1)
    n = int(node_ids.shape[0])
    if n == 0:
        return np.zeros((0, 8), dtype=np.float32)

    node_pos = {int(nid): i for i, nid in enumerate(node_ids.tolist())}
    feats = np.zeros((n, 8), dtype=np.float32)

    contact_nodes: set[int] = set()
    cpart = getattr(asm, "parts", {}).get("__CONTACT__")
    if cpart is not None:
        contact_nodes.update(int(v) for v in getattr(cpart, "node_ids", []) or [])
    for nid in contact_nodes:
        pos = node_pos.get(int(nid))
        if pos is not None:
            feats[pos, 0] = 1.0

    for bc in getattr(asm, "boundaries", []) or []:
        nid = _find_node_id_in_boundary_raw(getattr(bc, "raw", ""))
        if nid is None:
            continue
        pos = node_pos.get(int(nid))
        if pos is not None:
            feats[pos, 1] = 1.0
            ux, uy, uz = _find_boundary_dof_flags(getattr(bc, "raw", ""))
            feats[pos, 4] = max(feats[pos, 4], ux)
            feats[pos, 5] = max(feats[pos, 5], uy)
            feats[pos, 6] = max(feats[pos, 6], uz)

    mirror_nodes: set[int] = set()
    mirror_key = str(mirror_surface_name or "").strip().upper()
    for pname, part in getattr(asm, "parts", {}).items():
        if "MIRROR" in str(pname).upper() or (mirror_key and mirror_key in str(pname).upper()):
            mirror_nodes.update(int(v) for v in getattr(part, "node_ids", []) or [])
    for nid in mirror_nodes:
        pos = node_pos.get(int(nid))
        if pos is not None:
            feats[pos, 2] = 1.0

    # material tag as normalized scalar id (stable by sorted unique names)
    mat_lookup = {}
    for k, v in (part2mat or {}).items():
        mat_lookup[str(k).upper()] = str(v)
    mat_names = sorted(set(mat_lookup.values()))
    mat_to_id = {name: i for i, name in enumerate(mat_names)}
    denom = float(max(1, len(mat_names) - 1))
    # first-hit strategy if nodes belong to multiple parts
    for pname, part in getattr(asm, "parts", {}).items():
        up = str(pname).upper()
        mat_name = mat_lookup.get(up, None)
        if mat_name is None:
            continue
        mid = float(mat_to_id.get(mat_name, 0)) / denom
        for nid in getattr(part, "node_ids", []) or []:
            pos = node_pos.get(int(nid))
            if pos is None:
                continue
            if feats[pos, 3] == 0.0:
                feats[pos, 3] = np.float32(mid)

    feats[:, 7] = np.maximum.reduce([feats[:, 0], feats[:, 1], feats[:, 2]])

    return feats


def compute_uncertainty_proxy_sigma(
    u_pred: tf.Tensor,
    residual_scalar: tf.Tensor,
    *,
    proxy_scale: float = 1.0,
    eps: float = 1.0e-6,
) -> tf.Tensor:
    """Build residual-driven sigma proxy from predicted displacement magnitude."""

    u_pred = tf.cast(u_pred, tf.float32)
    residual_scalar = tf.cast(residual_scalar, tf.float32)
    umag = tf.sqrt(tf.reduce_sum(tf.square(u_pred), axis=1, keepdims=True) + tf.cast(eps, tf.float32))
    umag_mean = tf.reduce_mean(umag) + tf.cast(eps, tf.float32)
    sigma = tf.cast(proxy_scale, tf.float32) * residual_scalar * (umag / umag_mean) + tf.cast(eps, tf.float32)
    return sigma


def resolve_mixed_phase_flags(cfg: TrainerConfig) -> Dict[str, Any]:
    """Resolve mixed-bilevel phase switches from trainer config."""

    training_profile = str(getattr(cfg, "training_profile", "locked") or "locked").strip().lower().replace("-", "_")
    if training_profile not in {
        "strict_mixed_experimental",
        "strict_mixed_experimental_post_reentry",
        "normal_contact_first_mainline",
    }:
        return {
            "phase_name": "phase0",
            "normal_ift_enabled": False,
            "tangential_ift_enabled": False,
            "detach_inner_solution": True,
            "allow_full_ift_warmstart": False,
        }

    phase = getattr(cfg, "mixed_bilevel_phase", None)
    if phase is None:
        return {
            "phase_name": "phase0",
            "normal_ift_enabled": False,
            "tangential_ift_enabled": False,
            "detach_inner_solution": True,
            "allow_full_ift_warmstart": False,
        }
    return {
        "phase_name": str(getattr(phase, "phase_name", "phase0")),
        "normal_ift_enabled": bool(getattr(phase, "normal_ift_enabled", False)),
        "tangential_ift_enabled": bool(getattr(phase, "tangential_ift_enabled", False)),
        "detach_inner_solution": bool(getattr(phase, "detach_inner_solution", True)),
        "allow_full_ift_warmstart": bool(getattr(phase, "allow_full_ift_warmstart", False)),
    }


class Trainer(
    TrainerBuildMixin,
    TrainerPreloadMixin,
    TrainerInitMixin,
    TrainerMonitorMixin,
    TrainerOptMixin,
    TrainerRunMixin,
    TrainerVizMixin,
):
    @staticmethod
    def _build_node_semantic_features(
        asm: AssemblyModel,
        sorted_node_ids: np.ndarray,
        part2mat: Dict[str, str],
        mirror_surface_name: str,
    ) -> np.ndarray:
        return build_node_semantic_features(
            asm=asm,
            sorted_node_ids=sorted_node_ids,
            part2mat=part2mat,
            mirror_surface_name=mirror_surface_name,
        )

    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self._mixed_phase_flags = resolve_mixed_phase_flags(cfg)
        self._validate_locked_route()
        self._resolve_contact_backend()
        random.seed(cfg.seed)
        tf.keras.utils.set_random_seed(cfg.seed)
        np.random.seed(cfg.seed)
        tf.random.set_seed(cfg.seed)
        self._init_runtime_state(cfg)
        self._init_preload_sequence(cfg)
        self._init_device_and_precision(cfg)
        self._init_runtime_components()

    def _validate_locked_route(self) -> None:
        """Validate the only supported training route for this project."""

        cfg = self.cfg
        training_profile = str(getattr(cfg, "training_profile", "locked") or "locked").strip().lower().replace("-", "_")
        if training_profile in {
            "strict_mixed_experimental",
            "strict_mixed_experimental_post_reentry",
            "normal_contact_first_mainline",
            "p3_learning_gate",
        }:
            return
        stage_mode = str(getattr(getattr(cfg, "total_cfg", None), "preload_stage_mode", "") or "")
        stage_mode = stage_mode.strip().lower().replace("-", "_")
        violations: List[str] = []

        if not bool(getattr(cfg, "preload_use_stages", False)):
            violations.append("preload_use_stages must be true")
        if not bool(getattr(cfg, "incremental_mode", False)):
            violations.append("incremental_mode must be true")
        if bool(getattr(cfg, "stage_resample_contact", False)):
            violations.append("stage_resample_contact must be false")
        if int(getattr(cfg, "resample_contact_every", 0) or 0) > 0:
            violations.append("resample_contact_every must be <= 0")
        if bool(getattr(cfg, "contact_rar_enabled", False)):
            violations.append("contact_rar_enabled must be false")
        if bool(getattr(cfg, "volume_rar_enabled", False)):
            violations.append("volume_rar_enabled must be false")
        if bool(getattr(cfg, "lbfgs_enabled", False)):
            violations.append("lbfgs_enabled must be false")
        if bool(getattr(cfg, "friction_smooth_schedule", False)):
            violations.append("friction_smooth_schedule must be false")
        if bool(getattr(cfg, "viz_compare_cases", False)):
            violations.append("viz_compare_cases must be false")
        if stage_mode != "force_then_lock":
            violations.append("total_cfg.preload_stage_mode must be 'force_then_lock'")

        if violations:
            msg = "; ".join(violations)
            raise ValueError(f"[trainer] Locked route validation failed: {msg}")


    # ----------------- 杈呭姪宸ュ叿 -----------------
    def _cleanup_stale_ckpt_temp_dirs(self):
        ckpt_dir = getattr(self.cfg, "ckpt_dir", None)
        if not ckpt_dir:
            return
        try:
            entries = os.listdir(ckpt_dir)
        except Exception:
            return
        for name in entries:
            if not (name.startswith("ckpt-") and name.endswith("_temp")):
                continue
            path = os.path.join(ckpt_dir, name)
            try:
                shutil.rmtree(path, ignore_errors=True)
            except Exception:
                pass

    def _save_checkpoint_best_effort(self, checkpoint_number: Optional[int]) -> Optional[str]:
        if self.ckpt_manager is None:
            return None

        retries = max(0, int(getattr(self.cfg, "ckpt_save_retries", 0)))
        delay_s = float(getattr(self.cfg, "ckpt_save_retry_delay_s", 0.0))
        backoff = float(getattr(self.cfg, "ckpt_save_retry_backoff", 1.0))
        delay_s = max(0.0, delay_s)
        backoff = max(1.0, backoff)

        for attempt in range(retries + 1):
            try:
                # 鑻ュ榻?step 淇濆瓨澶辫触锛屽彲璁?manager 浣跨敤鍐呴儴鑷缂栧彿缁х画灏濊瘯锛?
                # 杩欐牱鑳介伩寮€鍚屽悕 *_temp 娈嬬暀瀵艰嚧鐨勪簩娆″け璐ャ€?
                if attempt == 0:
                    return self.ckpt_manager.save(checkpoint_number=checkpoint_number)
                return self.ckpt_manager.save(checkpoint_number=None)
            except UnicodeDecodeError as exc:
                msg = repr(exc)
                print(
                    f"[trainer] WARNING: checkpoint 淇濆瓨澶辫触 (UnicodeDecodeError) "
                    f"attempt={attempt + 1}/{retries + 1} ({msg})"
                )
            except Exception as exc:
                print(
                    f"[trainer] WARNING: checkpoint 淇濆瓨澶辫触 "
                    f"attempt={attempt + 1}/{retries + 1}: {exc}"
                )

            # 娓呯悊娈嬬暀 *_temp 鐩綍锛岄伩鍏嶄笅涓€娆′繚瀛樿鍚屽悕鐩綍褰卞搷
            self._cleanup_stale_ckpt_temp_dirs()
            if attempt < retries and delay_s > 0:
                time.sleep(delay_s)
                delay_s *= backoff
        return None

    @staticmethod
    def _wrap_bar_text(text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        return _wrap_white(str(text))

    def _set_pbar_desc(self, pbar, text: str) -> None:
        pbar.set_description_str(self._wrap_bar_text(text))

    def _set_pbar_postfix(self, pbar, text: str) -> None:
        if text is None:
            pbar.set_postfix_str(text)
            return
        pbar.set_postfix_str(self._wrap_bar_text(text))

    def _assemble_total(self) -> TotalEnergy:
        total = TotalEnergy(self.cfg.total_cfg)
        mixed_flags = dict(getattr(self, "_mixed_phase_flags", {}) or {})
        route_mode = self._resolve_bilevel_objective_route()
        protocol_settings = normal_contact_training_protocol.resolve_normal_contact_runtime_settings(
            self,
            route_mode=route_mode,
        )
        mixed_flags["contact_backend"] = self._resolve_contact_backend()
        mixed_flags["max_tail_qn_iters"] = int(protocol_settings.get("effective_max_tail_qn_iters", max(0, int(getattr(self.cfg, "max_tail_qn_iters", 0) or 0))))
        mixed_flags["coupling_phase_traction_scale"] = float(protocol_settings.get("coupling_phase_traction_scale", 1.0))
        mixed_flags["max_inner_iters_signature_gate"] = str(
            getattr(self.cfg, "max_inner_iters_signature_gate", "") or ""
        ).strip()
        mixed_flags["signature_gated_max_inner_iters"] = max(
            0,
            int(getattr(self.cfg, "signature_gated_max_inner_iters", 0) or 0),
        )
        mixed_flags["tangential_training_mode"] = str(
            getattr(self.cfg, "tangential_training_mode", "full") or "full"
        ).strip()
        mixed_flags["risk_guard_enabled"] = bool(getattr(self.cfg, "risk_guard_enabled", False))
        mixed_flags["risk_guard_scale"] = float(getattr(self.cfg, "risk_guard_scale", 1.0) or 1.0)
        total.set_mixed_bilevel_flags(mixed_flags)
        total.attach(
            elasticity=self.elasticity,
            contact=self.contact,
            tightening=self.tightening,
            bcs=self.bcs_ops,
        )
        return total

    # ----------------- Contact hardening schedule -----------------

    def _init_contact_hardening(self):
        """Initialise soft鈫抙ard schedule targets and apply soft start values."""

        if self.contact is None or not self.cfg.contact_hardening_enabled:
            self._contact_hardening_targets = None
            self._contact_hardening_frozen = False
            return

        self._contact_hardening_frozen = False

        def _to_float(x, fallback: float) -> float:
            try:
                if hasattr(x, "numpy"):
                    return float(x.numpy())
                return float(x)
            except Exception:
                return float(fallback)

        # Target (hard) values from config / operator
        beta_t = _to_float(getattr(self.contact.normal, "beta", None), self.cfg.contact_cfg.normal.beta)
        mu_n_t = _to_float(getattr(self.contact.normal, "mu_n", None), self.cfg.contact_cfg.normal.mu_n)
        k_t_t = _to_float(getattr(self.contact.friction, "k_t", None), self.cfg.contact_cfg.friction.k_t)
        mu_t_t = _to_float(getattr(self.contact.friction, "mu_t", None), self.cfg.contact_cfg.friction.mu_t)

        # Soft start values (user override or 20% of target)
        beta_s = float(self.cfg.contact_beta_start) if self.cfg.contact_beta_start is not None else 0.2 * beta_t
        mu_n_s = float(self.cfg.contact_mu_n_start) if self.cfg.contact_mu_n_start is not None else 0.2 * mu_n_t
        k_t_s = float(self.cfg.friction_k_t_start) if self.cfg.friction_k_t_start is not None else 0.2 * k_t_t
        mu_t_s = float(self.cfg.friction_mu_t_start) if self.cfg.friction_mu_t_start is not None else 0.2 * mu_t_t

        beta_s = max(beta_s, 1e-6)
        mu_n_s = max(mu_n_s, 1e-6)
        k_t_s = max(k_t_s, 0.0)
        mu_t_s = max(mu_t_s, 1e-6)

        # Apply soft start to operator variables
        try:
            self.contact.normal.beta.assign(beta_s)
            self.contact.normal.mu_n.assign(mu_n_s)
            self.contact.friction.k_t.assign(k_t_s)
            self.contact.friction.mu_t.assign(mu_t_s)
        except Exception:
            pass

        self._contact_hardening_targets = {
            "beta_start": beta_s,
            "beta_target": beta_t,
            "mu_n_start": mu_n_s,
            "mu_n_target": mu_n_t,
            "k_t_start": k_t_s,
            "k_t_target": k_t_t,
            "mu_t_start": mu_t_s,
            "mu_t_target": mu_t_t,
        }
        print(
            "[contact] soft鈫抙ard schedule init: "
            f"beta {beta_s:g}->{beta_t:g}, mu_n {mu_n_s:g}->{mu_n_t:g}, "
            f"k_t {k_t_s:g}->{k_t_t:g}, mu_t {mu_t_s:g}->{mu_t_t:g}"
        )

    def _maybe_update_contact_hardening(self, step: int):
        """Update contact penalty/ALM parameters according to training progress."""

        if self.contact is None:
            return

        if bool(getattr(self, "_strict_bilevel_backoff_requested", False)):
            try:
                current_fb_eps = float(getattr(self.contact.normal.cfg, "fb_eps", 1.0e-8) or 1.0e-8)
            except Exception:
                current_fb_eps = 1.0e-8
            softened_fb_eps = min(max(current_fb_eps * 2.0, 1.0e-8), 1.0e-3)
            try:
                self.contact.normal.cfg.fb_eps = softened_fb_eps
            except Exception:
                pass

            try:
                current_k_t = float(tf.cast(self.contact.friction.k_t, tf.float32).numpy())
            except Exception:
                current_k_t = None
            if current_k_t is not None:
                try:
                    self.contact.friction.set_k_t(max(current_k_t * 0.7, 1.0e-6))
                except Exception:
                    pass

            self._strict_bilevel_backoff_requested = False

        if self._contact_hardening_targets is None:
            return
        if bool(getattr(self, "_contact_hardening_frozen", False)):
            return
        if bool(getattr(self, "_strict_bilevel_freeze_requested", False)):
            self._contact_hardening_frozen = True
            self._continuation_freeze_events = int(getattr(self, "_continuation_freeze_events", 0) or 0) + 1
            return
        frac = float(np.clip(self.cfg.contact_hardening_fraction, 0.0, 1.0))
        if frac <= 0.0:
            return
        ramp_steps = max(1, int(frac * max(1, self.cfg.max_steps)))
        t = min(1.0, float(step) / float(ramp_steps))
        # Smooth cosine ramp
        s = 0.5 - 0.5 * math.cos(math.pi * t)

        def _lerp(a: float, b: float) -> float:
            return a + (b - a) * s

        beta = _lerp(self._contact_hardening_targets["beta_start"], self._contact_hardening_targets["beta_target"])
        mu_n = _lerp(self._contact_hardening_targets["mu_n_start"], self._contact_hardening_targets["mu_n_target"])
        k_t = _lerp(self._contact_hardening_targets["k_t_start"], self._contact_hardening_targets["k_t_target"])
        mu_t = _lerp(self._contact_hardening_targets["mu_t_start"], self._contact_hardening_targets["mu_t_target"])

        try:
            self.contact.normal.beta.assign(beta)
            self.contact.normal.mu_n.assign(mu_n)
            self.contact.friction.k_t.assign(k_t)
            self.contact.friction.mu_t.assign(mu_t)
        except Exception:
            pass

    def export_saved_model(self, export_dir: str) -> str:
        """Export the PINN displacement model as a TensorFlow SavedModel."""

        if self.model is None:
            raise RuntimeError("Trainer.export_saved_model() requires build()/restore().")

        n_bolts = max(1, len(self.cfg.preload_specs) or 3)
        stage_mode = str(getattr(self.cfg.total_cfg, "preload_stage_mode", "") or "")
        stage_mode = stage_mode.strip().lower().replace("-", "_")
        append_release_stage = bool(
            self.cfg.preload_use_stages
            and stage_mode == "force_then_lock"
            and bool(getattr(self.cfg, "preload_append_release_stage", True))
        )

        module = _SavedModelModule(
            model=self.model,
            use_stages=bool(self.cfg.preload_use_stages),
            append_release_stage=append_release_stage,
            shift=float(self.cfg.model_cfg.preload_shift),
            scale=float(self.cfg.model_cfg.preload_scale),
            n_bolts=n_bolts,
        )
        serving_fn = module.run.get_concrete_function()
        tf.saved_model.save(module, export_dir, signatures={"serving_default": serving_fn})
        print(f"[trainer] SavedModel exported -> {export_dir}")
        return export_dir

    # ----------------- 鍙鍖栵紙椴佹澶氱鍚嶏級 -----------------
