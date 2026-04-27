# -*- coding: utf-8 -*-
"""TrainerConfig extracted from trainer.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from model.pinn_model import ModelConfig
from physics.elasticity_config import ElasticityConfig
from physics.physical_scales import PhysicalScaleConfig
from physics.contact.contact_operator import ContactOperatorConfig
from physics.tightening_model import TighteningConfig
from model.loss_energy import TotalConfig


@dataclass
class MixedBilevelPhaseConfig:
    phase_name: str = "phase0"
    normal_ift_enabled: bool = False
    tangential_ift_enabled: bool = False
    detach_inner_solution: bool = True
    allow_full_ift_warmstart: bool = False


@dataclass
class SupervisionConfig:
    enabled: bool = False
    case_table_path: Optional[str] = None
    stage_dir: Optional[str] = None
    stage_count: int = 3
    single_case_id: Optional[str] = None
    selected_case_ids: Tuple[str, ...] = field(default_factory=tuple)
    single_case_stages: Tuple[int, ...] = field(default_factory=tuple)
    feature_mode: str = "cartesian"
    target_frame: str = "cartesian"
    annulus_center: Optional[Tuple[float, float]] = None
    annulus_r_in: Optional[float] = None
    annulus_r_out: Optional[float] = None
    annulus_fourier_order: int = 0
    split_group_key: str = "base_id"
    split_stratify_key: Optional[str] = "source"
    test_group_quotas: Dict[str, int] = field(
        default_factory=lambda: {
            "boundary": 1,
            "corner": 1,
            "interior": 3,
        }
    )
    cv_n_folds: int = 5
    cv_fold_index: int = 0
    train_splits: Tuple[str, ...] = ("train",)
    eval_splits: Tuple[str, ...] = ("val",)
    export_eval_reports: bool = True
    export_eval_plots: bool = True
    shuffle: bool = True
    seed: int = 42


@dataclass
class TwoStagePhaseConfig:
    max_steps: Optional[int] = None
    lr: Optional[float] = None
    save_best_on: Optional[str] = None
    validation_eval_every: Optional[int] = None
    supervision_contribution_floor_ratio: Optional[float] = None
    resume_ckpt_path: Optional[str] = None
    resume_start_step: int = 0
    resume_reset_optimizer: Optional[bool] = None
    base_weights: Dict[str, float] = field(default_factory=dict)
    mixed_bilevel_phase: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TwoStageTrainingConfig:
    enabled: bool = False
    phase1: TwoStagePhaseConfig = field(default_factory=TwoStagePhaseConfig)
    phase2: TwoStagePhaseConfig = field(default_factory=TwoStagePhaseConfig)


@dataclass
class TrainerConfig:
    inp_path: str = "data/shuangfan.inp"
    mirror_surface_name: str = "MIRROR up"
    mirror_surface_asm_key: Optional[str] = None

    materials: Dict[str, Any] = field(
        default_factory=lambda: {
            "mirror": (70000.0, 0.33),
            "steel": (210000.0, 0.30),
        }
    )
    part2mat: Dict[str, str] = field(
        default_factory=lambda: {
            "MIRROR": "mirror",
            "BOLT1": "steel",
            "BOLT2": "steel",
            "BOLT3": "steel",
        }
    )

    contact_pairs: List[Dict[str, str]] = field(default_factory=list)
    n_contact_points_per_pair: int = 6000
    contact_seed: int = 1234
    contact_two_pass: bool = False
    contact_mode: str = "sample"
    contact_mortar_gauss: int = 3
    contact_mortar_max_points: int = 0

    contact_hardening_enabled: bool = True
    contact_hardening_fraction: float = 0.4
    contact_beta_start: Optional[float] = None
    contact_mu_n_start: Optional[float] = None
    friction_k_t_start: Optional[float] = None
    friction_mu_t_start: Optional[float] = None

    preload_specs: List[Dict[str, str]] = field(default_factory=list)
    preload_n_points_each: int = 800

    bc_mode: str = "alm"
    bc_mu: float = 1.0e3
    bc_alpha: float = 1.0e4

    preload_min: float = 0.0
    preload_max: float = 2000.0
    preload_sequence: List[Any] = field(default_factory=list)
    preload_sequence_repeat: int = 1
    preload_sequence_shuffle: bool = False
    preload_sequence_jitter: float = 0.0

    preload_sampling: str = "lhs"
    preload_lhs_size: int = 64

    preload_use_stages: bool = True
    preload_randomize_order: bool = False
    preload_append_release_stage: bool = True

    incremental_mode: bool = True
    stage_inner_steps: int = 1
    stage_alm_every: int = 1
    reset_contact_state_per_case: bool = True
    stage_schedule_steps: List[int] = field(default_factory=list)
    training_profile: str = "locked"
    mixed_bilevel_phase: MixedBilevelPhaseConfig = field(default_factory=MixedBilevelPhaseConfig)
    contact_backend: str = "auto"
    max_tail_qn_iters: int = 0
    max_inner_iters_signature_gate: str = ""
    signature_gated_max_inner_iters: int = 0
    tangential_training_mode: str = "full"
    risk_guard_enabled: bool = False
    risk_guard_scale: float = 1.0
    risk_guard_allowed_buckets: Tuple[str, ...] = ("A", "B")
    protect_prefix_enabled: bool = False
    protect_first_n_steps: int = 0
    # Parked legacy control: kept for non-mainline scripts, not promoted YAML config.
    guard_activate_after_first_c_event: bool = False
    continuation_eps_shrink_cap: float = 0.7
    continuation_kt_growth_cap: float = 1.3

    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    elas_cfg: ElasticityConfig = field(
        default_factory=lambda: ElasticityConfig(coord_scale=1.0, chunk_size=0, use_pfor=False)
    )
    contact_cfg: ContactOperatorConfig = field(default_factory=ContactOperatorConfig)
    tightening_cfg: TighteningConfig = field(default_factory=TighteningConfig)
    physical_scales: PhysicalScaleConfig = field(default_factory=PhysicalScaleConfig)
    total_cfg: TotalConfig = field(
        default_factory=lambda: TotalConfig(
            w_int=1.0,
            w_cn=1.0,
            w_ct=1.0,
            w_bc=1.0,
            w_tight=1.0,
            w_sigma=1.0,
            w_eq=1.0,
            w_reg=1.0e-4,
        )
    )
    supervision: SupervisionConfig = field(default_factory=SupervisionConfig)
    two_stage_training: TwoStageTrainingConfig = field(default_factory=TwoStageTrainingConfig)
    resume_ckpt_path: Optional[str] = None
    resume_start_step: int = 0
    resume_reset_optimizer: bool = False
    trainable_scope: str = "all"
    ase_residual_warmup_steps: int = 0
    run_phase_name: Optional[str] = None

    loss_adaptive_enabled: bool = True
    loss_update_every: int = 1
    loss_ema_decay: float = 0.95
    loss_min_factor: float = 0.25
    loss_max_factor: float = 4.0
    loss_min_weight: Optional[float] = None
    loss_max_weight: Optional[float] = None
    loss_gamma: float = 2.0
    loss_focus_terms: Tuple[str, ...] = field(default_factory=tuple)
    supervision_contribution_floor_enabled: bool = False
    supervision_contribution_floor_ratio: float = 0.0

    max_steps: int = 1000
    adam_steps: Optional[int] = None
    lr: float = 1e-3
    grad_clip_norm: Optional[float] = 1.0
    log_every: int = 1
    validation_eval_every: int = 0
    alm_update_every: int = 0
    early_exit_enabled: bool = True
    early_exit_warmup_steps: int = 200
    early_exit_nonfinite_patience: int = 8
    early_exit_divergence_patience: int = 30
    early_exit_grad_norm_threshold: float = 1.0e6
    early_exit_pi_ema_rel_increase: float = 0.5
    early_exit_check_every: int = 1
    contact_route_update_every: int = 1
    uncertainty_loss_weight: float = 0.0
    uncertainty_sample_points: int = 0
    uncertainty_proxy_scale: float = 1.0
    uncertainty_logvar_min: float = -8.0
    uncertainty_logvar_max: float = 6.0
    val_plateau_lr_decay_enabled: bool = False
    val_plateau_lr_decay_metric: str = "val_drrms"
    val_plateau_lr_decay_warmup: int = 0
    val_plateau_lr_decay_patience: int = 0
    val_plateau_lr_decay_factor: float = 0.5
    val_plateau_lr_decay_min_lr: float = 1.0e-6

    build_bar_color: Optional[str] = "cyan"
    train_bar_color: Optional[str] = "cyan"
    step_bar_color: Optional[str] = "green"
    build_bar_enabled: bool = True
    train_bar_enabled: bool = True
    step_bar_enabled: bool = False
    tqdm_disable: bool = False
    tqdm_disable_if_not_tty: bool = True

    mixed_precision: Optional[str] = "mixed_float16"
    seed: int = 42

    out_dir: str = "outputs"
    ckpt_dir: str = "checkpoints"
    ckpt_max_to_keep: int = 3
    ckpt_save_retries: int = 3
    ckpt_save_retry_delay_s: float = 1.0
    ckpt_save_retry_backoff: float = 2.0
    graph_cache_enabled: bool = True
    graph_cache_dir: Optional[str] = None
    graph_cache_name: Optional[str] = None
    viz_samples_after_train: int = 6
    viz_title_prefix: str = "Total Deformation (trained PINN)"
    viz_style: str = "smooth"
    viz_colormap: str = "turbo"
    viz_diagnose_blanks: bool = False
    viz_auto_fill_blanks: bool = False
    viz_levels: int = 64
    viz_symmetric: bool = False
    viz_units: str = "mm"
    viz_draw_wireframe: bool = False
    viz_surface_enabled: bool = True
    viz_surface_source: str = "part_top"
    viz_write_data: bool = True
    viz_write_surface_mesh: bool = False
    viz_plot_full_structure: bool = False
    viz_full_structure_part: Optional[str] = "mirror1"
    viz_write_full_structure_data: bool = False
    viz_retriangulate_2d: bool = False
    viz_refine_subdivisions: int = 3
    viz_refine_max_points: int = 180_000
    viz_use_shape_function_interp: bool = False
    viz_smooth_vector_iters: int = 0
    viz_smooth_vector_lambda: float = 0.35
    viz_smooth_scalar_iters: int = 0
    viz_smooth_scalar_lambda: float = 0.6
    viz_eval_batch_size: int = 65_536
    viz_eval_scope: str = "assembly"
    viz_force_pointwise: bool = False
    viz_remove_rigid: bool = True
    viz_use_last_training_case: bool = False
    viz_write_reference_aligned: bool = True
    viz_reference_truth_path: Optional[str] = "auto"
    viz_plot_stages: bool = False
    viz_skip_release_stage_plot: bool = False
    viz_compare_cmap: str = "coolwarm"
    viz_compare_common_scale: bool = True
    viz_supervision_compare_enabled: bool = False
    viz_supervision_compare_split: str = "test"
    viz_supervision_compare_sources: Tuple[str, ...] = ("boundary", "corner", "interior")
    viz_same_pipeline_supervision_debug: bool = False
    viz_export_final_and_best: bool = False
    save_best_on: str = "Pi"

    yield_strength: Optional[float] = None
