from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from typing import Any

from regression.src.train.train_retrain import generate_regression_training_data
from regression.src.unlearn.unlearn_regression import unlearn
from regression.src.ripple.ripple_regression import (
    evaluate_models_by_ripple,
    aggregate_by_bin,
)
from regression.src.utils import set_seed, mse_loss


def _build_setting(cfg):
    rng = set_seed(cfg.seed)
    task_type = "regression" if cfg.model_type == "linear" else "classification"
    setting = generate_regression_training_data(cfg, rng, task_type)
    return setting, task_type


def _evaluate_single_unlearn_on_fixed_setting(
    cfg,
    setting,
    task_type: str,
    unlearn_seed: int,
):
    theta_unlearn, _, _ = unlearn(cfg, setting, unlearn_seed=unlearn_seed)

    bin_edges = np.linspace(cfg.rho_min, cfg.rho_max, cfg.n_bins)

    eval_result = evaluate_models_by_ripple(
        X_test=setting["X_test"],
        y_test=(
            setting["y_test"]
            if cfg.model_type == "linear"
            else setting["y_test_logits"]
        ),
        x_a=setting["x_a"],
        models={
            "retrain": setting["theta_retrain"],
            cfg.unlearning_method: theta_unlearn,
        },
        bin_edges=bin_edges,
        task_type=task_type,
    )

    return eval_result


def run_multiple_experiments_fixed_interval_for_variance_plot(
    cfg,
    n_outer_runs=30,
    n_inner_unlearn_runs=50,
):
    """
    Correct variance/flip pipeline for RL:
    - outer loop: different deleted points / datasets
    - inner loop: repeated stochastic unlearning on the same fixed setting
    """
    if not (cfg.model_type == "logistic" and cfg.unlearning_method == "rl"):
        raise ValueError("This function is intended only for logistic + rl.")

    all_var_curves = []
    all_flip_prob_curves = []
    last_setting = None

    bin_edges = np.linspace(cfg.rho_min, cfg.rho_max, cfg.n_bins)
    n_bins = len(bin_edges) - 1

    for outer_idx in tqdm(range(n_outer_runs)):
        cfg_outer = deepcopy(cfg)
        cfg_outer.seed = cfg.seed + outer_idx

        setting, task_type = _build_setting(cfg_outer)
        last_setting = setting

        delta_logit_runs = []
        flip_indicator_runs = []
        bin_idx_ref = None

        for inner_idx in range(n_inner_unlearn_runs):
            unlearn_seed = 10_000 * outer_idx + inner_idx

            eval_result = _evaluate_single_unlearn_on_fixed_setting(
                cfg=cfg_outer,
                setting=setting,
                task_type=task_type,
                unlearn_seed=unlearn_seed,
            )

            delta_logit_runs.append(eval_result["rl_delta_logit"])
            flip_indicator_runs.append(eval_result["rl_flip_indicator"])

            if bin_idx_ref is None:
                bin_idx_ref = eval_result["bin_idx"]

        delta_logit_runs = np.stack(delta_logit_runs, axis=0)
        flip_indicator_runs = np.stack(flip_indicator_runs, axis=0)

        # Variance across stochastic unlearning runs, for each fixed x_q
        pointwise_var = np.var(delta_logit_runs, axis=0)

        # Flip probability across stochastic unlearning runs, for each fixed x_q
        pointwise_flip_prob = np.mean(flip_indicator_runs, axis=0)

        var_by_bin, _ = aggregate_by_bin(pointwise_var, bin_idx_ref, n_bins)
        flip_prob_by_bin, _ = aggregate_by_bin(pointwise_flip_prob, bin_idx_ref, n_bins)

        all_var_curves.append(var_by_bin)
        all_flip_prob_curves.append(flip_prob_by_bin)

    all_var_curves = np.array(all_var_curves)
    all_flip_prob_curves = np.array(all_flip_prob_curves)

    stats = {
        "delta_logit_var_fixed_xa": {
            "mean": np.nanmean(all_var_curves, axis=0),
            "std": np.nanstd(all_var_curves, axis=0),
            "stderr": np.nanstd(all_var_curves, axis=0)
            / np.sqrt(all_var_curves.shape[0]),
            "all_runs": all_var_curves,
        },
        "flip_prob_fixed_xa": {
            "mean": np.nanmean(all_flip_prob_curves, axis=0),
            "std": np.nanstd(all_flip_prob_curves, axis=0),
            "stderr": np.nanstd(all_flip_prob_curves, axis=0)
            / np.sqrt(all_flip_prob_curves.shape[0]),
            "all_runs": all_flip_prob_curves,
        },
    }

    return stats, last_setting


def run_multiple_experiments_for_variance_plot(
    cfg,
    n_outer_runs=30,
    n_inner_unlearn_runs=50,
):
    stats_full = {}
    setting_full = {}

    for confidence_level in cfg.confidence_levels:
        cfg.confidence_level = confidence_level

        stats, setting = run_multiple_experiments_fixed_interval_for_variance_plot(
            cfg=cfg,
            n_outer_runs=n_outer_runs,
            n_inner_unlearn_runs=n_inner_unlearn_runs,
        )

        stats_full[confidence_level] = stats
        setting_full[confidence_level] = setting

    return stats_full, setting_full


def run_experiment(cfg) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setting, task_type = _build_setting(cfg)

    theta_unlearn, trajectory, del_losses = unlearn(cfg, setting)

    setting["theta_unlearn"] = theta_unlearn

    # Evaluate by ripple bins
    bin_edges = np.linspace(cfg.rho_min, cfg.rho_max, cfg.n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    eval_result = evaluate_models_by_ripple(
        X_test=setting["X_test"],
        y_test=(
            setting["y_test"]
            if cfg.model_type == "linear"
            else setting["y_test_logits"]
        ),
        x_a=setting["x_a"],
        models={
            "full": setting["theta_full"],
            "retrain": setting["theta_retrain"],
            cfg.unlearning_method: theta_unlearn,
        },
        bin_edges=bin_edges,
        task_type="regression" if cfg.model_type == "linear" else "classification",
    )

    sq_logit_diff_by_bin = eval_result[f"{cfg.unlearning_method}_sq_logit_diff_by_bin"]
    mse_loss_diff_by_bin =  eval_result[f"{cfg.unlearning_method}_mse_loss_by_bin"]
    # delta_logit_by_bin = eval_result[f"{cfg.unlearning_method}_delta_logit_by_bin"]
    # delta_logit_var_by_bin = eval_result[
    #     f"{cfg.unlearning_method}_delta_logit_var_by_bin"
    # ]
    full_sq_logit_diff_by_bin = eval_result["full_sq_logit_diff_by_bin"]
    full_mse_loss_diff_by_bin = eval_result["full_mse_loss_by_bin"]
    # pointwise_sq_delta_logit = eval_result[f"{cfg.unlearning_method}_sq_delta_logit"]
    # pointwise_delta_logit = eval_result[f"{cfg.unlearning_method}_delta_logit"]

    # if cfg.model_type == "logistic":
    #     flip_rate_by_bin = eval_result[f"{cfg.unlearning_method}_flip_rate_by_bin"]
    #     pointwise_flip_indicator = eval_result[
    #         f"{cfg.unlearning_method}_flip_indicator"
    #     ]
    #     full_flip_rate_by_bin = eval_result["full_flip_rate_by_bin"]

    deleted_sq_logit_diff = float(
        setting["x_a"] @ theta_unlearn - setting["x_a"] @ setting["theta_retrain"]
    )**2

    x_a, y_a =  setting["x_a"], setting["y_a"]
    deleted_mse_loss_diff =  mse_loss(y_a, x_a @ theta_unlearn) - mse_loss(y_a, x_a @ setting["theta_retrain"])

    # Save raw arrays / metadata
    summary = {
        "config": asdict(cfg),
        "deleted_point_norm_sq": float(setting["x_a"] @ setting["x_a"]),
        "deleted_point_logit": float(setting["deleted_point_logit"]),
        "theta_full_norm": float(np.linalg.norm(setting["theta_full"])),
        "theta_retrain_norm": float(np.linalg.norm(setting["theta_retrain"])),
        "theta_unlearn_norm": float(np.linalg.norm(theta_unlearn)),
        "param_distance_full_vs_retrain": float(
            np.linalg.norm(setting["theta_full"] - setting["theta_retrain"])
        ),
        "param_distance_unlearn_vs_retrain": float(
            np.linalg.norm(theta_unlearn - setting["theta_retrain"])
        ),
        "bin_centers": bin_centers.tolist(),
        "deleted_point_loss_trajectory": del_losses,
        "rho": eval_result["rho"].tolist(),
        "bin_idx": eval_result["bin_idx"].tolist(),
        "sq_logit_diff_by_bin": sq_logit_diff_by_bin.tolist(),
        "mse_loss_diff_by_bin": mse_loss_diff_by_bin.tolist(),
        # "delta_logit_by_bin": delta_logit_by_bin.tolist(),
        # "delta_logit_var_by_bin": delta_logit_var_by_bin.tolist(),
        "full_sq_logit_diff_by_bin": full_sq_logit_diff_by_bin.tolist(),
        "full_mse_loss_diff_by_bin": full_mse_loss_diff_by_bin.tolist(),
        # "pointwise_sq_delta_logit": pointwise_sq_delta_logit.tolist(),
        # "pointwise_delta_logit": pointwise_delta_logit.tolist(),
        # "deleted_delta_logit": deleted_delta_logit,
        "deleted_sq_logit_diff": deleted_sq_logit_diff,
        "deleted_mse_loss_diff":  deleted_mse_loss_diff,
        "deleted_rho": 1.0,
    }

    # if cfg.model_type == "logistic":
    #     summary["flip_rate_by_bin"] = flip_rate_by_bin.tolist()
    #     summary["full_flip_rate_by_bin"] = full_flip_rate_by_bin.tolist()
    #     summary["pointwise_flip_indicator"] = pointwise_flip_indicator.tolist()

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary, setting


def _nan_stats(runs: list[Any]) -> dict[str, Any]:
    data = np.asarray(runs, dtype=float)

    return {
        "mean": np.nanmean(data, axis=0),
        "std": np.nanstd(data, axis=0),
        "stderr": np.nanstd(data, axis=0) / np.sqrt(data.shape[0]),
        "all_runs": data,
    }

def run_multiple_experiments_fixed_interval(cfg, n_runs: int = 30):
    """
    Run the ripple experiment multiple times with different seeds and aggregate statistics.
    """
    metric_keys = {
        "sq_logit_diff": "sq_logit_diff_by_bin",
        "mse_loss_diff": "mse_loss_diff_by_bin",
        "full_sq_logit_diff": "full_sq_logit_diff_by_bin",
        "full_mse_loss_diff": "full_mse_loss_diff_by_bin",
        "deleted_point_sq_logit_diff": "deleted_sq_logit_diff",
        "deleted_point_mse_loss_diff": "deleted_mse_loss_diff",
    }

    runs = {name: [] for name in metric_keys}
    setting = None

    for run_idx in tqdm(range(n_runs)):
        cfg_i = deepcopy(cfg)
        cfg_i.seed = cfg.seed + run_idx

        result, setting = run_experiment(cfg_i)

        for output_name, result_key in metric_keys.items():
            runs[output_name].append(result[result_key])

    stats = {
        name: _nan_stats(values)
        for name, values in runs.items()
    }

    return stats, setting


def run_multiple_experiments(cfg, n_runs=30):

    stats_full = {}
    setting_full = {}

    for confidence_level in cfg.confidence_levels:

        cfg.confidence_level = confidence_level

        stats, setting = run_multiple_experiments_fixed_interval(cfg, n_runs=n_runs)

        stats_full[confidence_level] = stats
        setting_full[confidence_level] = setting

    return stats_full, setting_full
