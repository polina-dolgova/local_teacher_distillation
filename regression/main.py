from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import numpy as np
import time

from regression.src.draw_graphs import (
    plot_graphs,
    plot_rl_flip_and_variance_fixed_xa,
    plot_sq_logit_diff_comparison,
)
from regression.src.ripple_experiment import (
    run_multiple_experiments,
    run_multiple_experiments_for_variance_plot,
)


@dataclass
class ExperimentConfig:
    # Data dimensions
    n_train: int = 400
    n_test: int = 30000
    n_delete: int = 1

    n_ft: int = 0
    n_support: int = 0

    d: int = 200
    h: int = 12

    # Data / noise
    train_noise_std: float = 0.1
    test_noise_std: float = 0.0

    # Linear regression regularization
    ridge: float = 1e-6

    # Gradient ascent unlearning
    unlearn_steps: int = 80
    unlearn_lr: float = 0.05

    # Ripple bins
    rho_min: float = -1.0
    rho_max: float = 1.0
    n_bins: int = 31

    # Randomness
    seed: int = 42
    seed_deleted_point: int = 43

    # Deletion point selection
    deleted_index: int | None = None

    # Saving
    out_dir: str = "ripple_outputs"

    model_type: str = "linear"
    unlearning_method: str = "ga"

    confidence_split: bool = False
    confidence_levels: tuple = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ripple-effect experiment for linear regression unlearning."
    )
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--d", type=int, default=10)

    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--n-delete", type=int, default=1)

    parser.add_argument("--n-train-rl", type=int, default=900)
    parser.add_argument("--d-rl", type=int, default=1000)
    
    parser.add_argument("--h", type=int, default=12)
    parser.add_argument("--n-ft", type=int, default=0)
    parser.add_argument("--n-support", type=int, default=0)

    parser.add_argument("--train-noise-std", type=float, default=0.1)
    parser.add_argument("--test-noise-std", type=float, default=0.0)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--unlearn-steps", type=int, default=500)
    parser.add_argument("--unlearn-lr", type=float, default=0.01)
    parser.add_argument("--rho-min", type=float, default=-1.0)
    parser.add_argument("--rho-max", type=float, default=1.0)
    parser.add_argument("--n-bins", type=int, default=30)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--deleted-index", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="regression/ripple_outputs")
    parser.add_argument(
        "--model_type", type=str, default="linear", choices=["linear", "logistic"]
    )
    parser.add_argument(
        "--unlearning_method", type=str, default="ga", choices=["ga", "rl"]
    )
    parser.add_argument("--seed-deleted-point", type=int, default=43)
    parser.add_argument("--confidence-split", type=bool, default=False)

    parser.add_argument("--compare-ga-rl", action="store_true")
    parser.add_argument("--ga-unlearn-steps", type=int, default=None)
    parser.add_argument("--ga-unlearn-lr", type=float, default=None)
    parser.add_argument("--rl-unlearn-steps", type=int, default=None)
    parser.add_argument("--rl-unlearn-lr", type=float, default=None)

    parser.add_argument("--n-inner-unlearn-runs", type=int, default=50)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    timestamp = int(time.time())

    output_dir = (
        f"{args.out_dir}/{args.model_type}/{args.unlearning_method}/{timestamp}"
    )

    base_cfg = ExperimentConfig(
        n_train=args.n_train,
        n_test=args.n_test,
        n_delete=args.n_delete,
        n_ft=args.n_ft,
        n_support=args.n_support,
        d=args.d,
        h=args.h,
        train_noise_std=args.train_noise_std,
        test_noise_std=args.test_noise_std,
        ridge=args.ridge,
        unlearn_steps=args.unlearn_steps,
        unlearn_lr=args.unlearn_lr,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        n_bins=args.n_bins,
        seed=args.seed,
        deleted_index=args.deleted_index,
        out_dir=output_dir,
        model_type=args.model_type,
        unlearning_method=args.unlearning_method,
        seed_deleted_point=args.seed_deleted_point,
        confidence_split=args.confidence_split,
        confidence_levels=["low", "mid", "high"] if args.confidence_split else [None],
    )

    N_RUNS = 100
    bin_edges = np.linspace(base_cfg.rho_min, base_cfg.rho_max, base_cfg.n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if args.compare_ga_rl:
        compare_dir = f"{args.out_dir}/{args.model_type}/ga_vs_rl/{timestamp}"

        ga_cfg = replace(
            base_cfg,
            out_dir=f"{compare_dir}/ga",
            unlearning_method="ga",
            model_type="linear",
            unlearn_steps=(
                args.ga_unlearn_steps
                if args.ga_unlearn_steps is not None
                else args.unlearn_steps
            ),
            unlearn_lr=(
                args.ga_unlearn_lr
                if args.ga_unlearn_lr is not None
                else args.unlearn_lr
            ),
        )
        rl_cfg = replace(
            base_cfg,
            out_dir=f"{compare_dir}/rl",
            unlearning_method="rl",
            model_type="logistic",
            unlearn_steps=(
                args.rl_unlearn_steps
                if args.rl_unlearn_steps is not None
                else args.unlearn_steps
            ),
            unlearn_lr=(
                args.rl_unlearn_lr
                if args.rl_unlearn_lr is not None
                else args.unlearn_lr
            ),
            n_train=args.n_train_rl,
            d=args.d_rl
        )

        ga_stats_full, ga_setting_full = run_multiple_experiments(ga_cfg, n_runs=N_RUNS)
        rl_stats_full, rl_setting_full = run_multiple_experiments(rl_cfg, n_runs=N_RUNS)

        plot_sq_logit_diff_comparison(
            method_results={
                "GA": (ga_cfg, ga_stats_full, ga_setting_full),
                "RL": (rl_cfg, rl_stats_full, rl_setting_full),
            },
            bin_centers=bin_centers,
            out_path=f"{compare_dir}/sq_logit_diff_ga_vs_rl.pdf",
        )
        return

    # stats_full, setting_full = run_multiple_experiments(base_cfg, n_runs=N_RUNS)
    # plot_graphs(base_cfg, bin_centers, stats_full, setting_full)

    # if base_cfg.model_type == "logistic" and base_cfg.unlearning_method == "rl":
    #     variance_stats_full, _ = run_multiple_experiments_for_variance_plot(
    #         base_cfg,
    #         n_outer_runs=N_RUNS,
    #         n_inner_unlearn_runs=args.n_inner_unlearn_runs,
    #     )

    #     plot_rl_flip_and_variance_fixed_xa(
    #         base_cfg,
    #         bin_centers,
    #         variance_stats_full,
    #         f"{base_cfg.out_dir}/rl_flip_and_variance_fixed_xa.pdf",
    #     )


if __name__ == "__main__":
    main()
