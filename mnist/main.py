from __future__ import annotations

from mnist.src.plot import plot_all_similarity_heatmaps
from mnist.src.evaluate import compare_models
from mnist.src.unlearn.unlearn_random_labels import unlearn_class
from mnist.src.unlearn.unlearn_distill import unlearn_with_distilled_labels
from mnist.src.train import train_models
from mnist.src.utils import (
    get_default_device,
    parse_config,
    build_output_dir,
    save_config,
)
from mnist.src.unlearn.unlearn_ga import unlearn_class_with_ga


def main(cfg, output_dir, device="cpu"):
    model, retrain_model = train_models(
        class_to_forget=cfg.class_to_forget,
        device=device,
        seed=cfg.seed,
    )


    unlearned_models = {"Gradient Ascent": [], "Random Labels": []}
    similarity_matrices = []

    for n in range(cfg.n_random_repeats):
        unlearned_model_rl = unlearn_class(
            model,
            device=device,
            class_to_forget=cfg.class_to_forget,
            steps=cfg.unlearn_steps,
            lr=cfg.unlearn_lr,
            weight_decay=cfg.unlearn_weight_decay,
            seed=cfg.seed + n,  # Different seed for each repeat
        )

        unlearned_model_ga = unlearn_class_with_ga(
            full_model=model,
            class_to_forget=cfg.class_to_forget,
            steps=cfg.unlearn_steps_ga,
            lr=cfg.unlearn_lr_ga,
            weight_decay=cfg.unlearn_weight_decay,
            device=device,
            seed=cfg.seed + n
        )

        unlearned_models["Gradient Ascent"].append(unlearned_model_ga)
        unlearned_models["Random Labels"].append(unlearned_model_rl)

    similarity_matrices = plot_all_similarity_heatmaps(
        output_dir=output_dir / "heatmaps",
        model=model,
        unlearned_model=unlearned_models["Gradient Ascent"][0],
        device=device,
    )

    compare_models(
        model=retrain_model,
        unlearned_models_dict=unlearned_models,
        class_to_forget=cfg.class_to_forget,
        similarity_matrices=similarity_matrices,
        output_dir=output_dir / "classical_rl_ga",
        device=device,
    )

    for support_selection in cfg.support_selections:
        unlearn_with_distilled_labels(
            model=model,
            retrain_model=retrain_model,
            output_dir=output_dir,
            device=device,
            similarity_matrices=similarity_matrices,
            class_to_forget=cfg.class_to_forget,
            support_size=cfg.support_size,
            support_selection=support_selection,
            support_feature_space=cfg.support_feature_space,
            teacher_epochs=cfg.teacher_epochs,
            teacher_batch_size=cfg.teacher_batch_size,
            teacher_lr=cfg.teacher_lr,
            teacher_weight_decay=cfg.teacher_weight_decay,
            unlearn_steps=cfg.unlearn_steps,
            unlearn_batch_size=cfg.unlearn_batch_size,
            unlearn_lr=cfg.unlearn_lr,
            unlearn_weight_decay=cfg.unlearn_weight_decay,
            label_mode=cfg.label_mode,
            k_neighbor=cfg.k_neighbor,
            teacher_selection=cfg.teacher_selection,
            temperature=cfg.temperature,
            zero_forget_class_prob=cfg.zero_forget_class_prob,
            seed=cfg.seed,
        )


if __name__ == "__main__":
    cfg = parse_config()
    device = get_default_device()
    output_dir = build_output_dir(cfg)
    save_config(cfg, output_dir)
    main(cfg=cfg, output_dir=output_dir, device=device)
