import torch
from pathlib import Path
from torch.utils.data import Subset

from cifar.src.custom_types import *
from cifar.src.datasets import (
    get_dataset_class_names,
    build_cifar_dataset,
    split_indices_by_forget_class,
)
from cifar.src.models.backbones import get_model
from cifar.src.models.features import get_feature_extractor
from cifar.src.evaluation.accuracy import compute_per_class_accuracy
from cifar.src.evaluation.similarity import (
    compute_cosine_similarity_to_forget_class,
    compute_imagenet_class_mean_embeddings_cifar,
    compute_full_model_class_mean_embeddings_cifar,
)

from cifar.src.visualization.plots import (
    save_per_class_accuracy_bar_chart,
    save_accuracy_drop_vs_similarity_plot,
)
from cifar.src.config import (
    ExperimentConfig,
    save_experiment_config,
    copy_unlearning_config_dir,
)
from cifar.src.evaluation.metrics import collect_extra_metrics
from cifar.src.evaluation.svc_mia import compute_svc_mia_forget_efficacy
from cifar.methods.distill.teacher_metrics import collect_teacher_metrics


def evaluate_model_accuracy(
    model_path,
    cfg: ExperimentConfig,
    retain_dataset,
    forget_dataset,
    class_names,
    mode="before_unlearning",
    evaluate=True,
    device="cpu",
) -> None:

    model = get_model(
        dataset_name=cfg.dataset_name, device=device, model_path=model_path
    )

    if not evaluate:
        return model, None, None

    # Post-unlearning accuracy
    retain_accuracy, forget_accuracy = compute_accuracy(
        model=model,
        retain_dataset=retain_dataset,
        forget_dataset=forget_dataset,
        cfg=cfg,
        class_names=class_names,
        mode=mode,
        device=device,
    )

    return model, retain_accuracy, forget_accuracy


def compute_accuracy(
    model: torch.nn.Module,
    retain_dataset: torch.utils.data.Dataset,
    forget_dataset: torch.utils.data.Dataset | None,
    cfg: ExperimentConfig,
    class_names: list[str],
    mode: str = "before_unlearning",
    device="cpu",
):
    num_classes = len(class_names)
    retain_accuracy = compute_per_class_accuracy(
        model,
        retain_dataset,
        num_classes=num_classes,
        device=device,
    )

    forget_accuracy = compute_per_class_accuracy(
        model,
        forget_dataset,
        num_classes=num_classes,
        device=device,
    )

    output_path = cfg.output_dir / f"per_class_accuracy_{mode}.png"

    save_per_class_accuracy_bar_chart(
        retain_per_class_acc=retain_accuracy,
        forget_per_class_acc=forget_accuracy,
        class_names=class_names,
        output_path=output_path,
        title=f"Per-class accuracy {mode} ({cfg.dataset_name})",
        selected_classes=cfg.selected_classes,
    )

    return retain_accuracy, forget_accuracy


def compute_class_mean_embeddings(
    full_model, retain_dataset, forget_dataset, cfg, device="cpu"
):
    if cfg.similarity_source == "imagenet_resnet18":
        class_mean_embeddings_retain = compute_imagenet_class_mean_embeddings_cifar(
            dataset=retain_dataset,
            num_classes=cfg.num_classes,
            selected_classes=cfg.selected_classes,
            device=device,
        )
        class_mean_embeddings_forget = compute_imagenet_class_mean_embeddings_cifar(
            dataset=forget_dataset,
            num_classes=cfg.num_classes,
            selected_classes=cfg.selected_classes,
            device=device,
        )

        similarity_xlabel = (
            "Cosine similarity to forget class (ImageNet ResNet-18 embedding)"
        )

    elif cfg.similarity_source == "full_model":
        feature_extractor_fn = get_feature_extractor(cfg.dataset_name)
        class_mean_embeddings_retain = compute_full_model_class_mean_embeddings_cifar(
            model=full_model,
            dataset=retain_dataset,
            num_classes=cfg.num_classes,
            selected_classes=cfg.selected_classes,
            device=device,
            feature_extractor_fn=feature_extractor_fn,
        )
        class_mean_embeddings_forget = compute_full_model_class_mean_embeddings_cifar(
            model=full_model,
            dataset=forget_dataset,
            num_classes=cfg.num_classes,
            selected_classes=[cfg.class_to_forget],
            device=device,
            feature_extractor_fn=feature_extractor_fn,
        )
        similarity_xlabel = "Cosine similarity to forget class (full model embedding)"

    else:
        raise ValueError(f"Unknown similarity_source: {cfg.similarity_source}")

    return class_mean_embeddings_retain, class_mean_embeddings_forget, similarity_xlabel


def run_experiment(cfg: ExperimentConfig, device="cpu") -> None:
    """
    Full pipeline:
    1) Load pretrained CIFAR-10 ResNet-56
    2) Save pre-unlearning per-class accuracy
    3) Apply unlearning to one class
    4) Save post-unlearning per-class accuracy
    5) Save accuracy_drop vs cosine_similarity plot
    """

    print(f"Using device {device}....")
    class_names = get_dataset_class_names(cfg.dataset_name)
    cfg.num_classes = len(class_names)
    if cfg.selected_classes is None:
        cfg.selected_classes = list(range(cfg.num_classes))

    if cfg.test_on_train:
        print("Evaluating on train set...")
        dataset = build_cifar_dataset(
            dataset_name=cfg.dataset_name, train=True, do_transforms=False
        )
        forget_indices, retain_indices = split_indices_by_forget_class(
            dataset, cfg.class_to_forget, cfg.class_fraction_to_forget
        )

        retain_dataset = Subset(dataset, retain_indices)
        forget_dataset = Subset(dataset, forget_indices)
        print(len(retain_dataset), len(forget_dataset))
    else:
        retain_dataset = build_cifar_dataset(
            dataset_name=cfg.dataset_name,
            train=False,
            selected_classes=cfg.selected_classes,
        )

        # what we really have unlearned
        train_dataset = build_cifar_dataset(
            dataset_name=cfg.dataset_name, train=True, do_transforms=False
        )
        forget_indices, _ = split_indices_by_forget_class(
            train_dataset, cfg.class_to_forget, cfg.class_fraction_to_forget
        )
        forget_dataset = Subset(train_dataset, forget_indices)

    test_dataset = build_cifar_dataset(dataset_name=cfg.dataset_name, train=False)
    ## FULLY TRAINED MODEL ###
    print("Evaluating full model accuracy...")
    full_model = get_model(
        dataset_name=cfg.dataset_name,
        device=device,
        model_path=cfg.full_model_path,
    )

    if cfg.teacher_model_path is not None:
        print("calculating teacher metrics....")
        cfg.teacher_metrics = collect_teacher_metrics(
            unlearning_config_path=cfg.unlearning_config_path,
            teacher_model_path=cfg.teacher_model_path,
            dataset_name=cfg.dataset_name,
            feature_extractor=full_model,
            forget_dataset=forget_dataset,
            num_classes=cfg.num_classes,
            device=device,
            class_to_forget=cfg.class_to_forget,
        )

    # ### RETRAIN MODEL ###
    # print("Evaluating retrain model accuracy...")
    # retrain_models = []

    # for retrain_model_path in cfg.retrain_model_paths:
    #     retrain_model = get_cifar_resnet56(
    #         dataset_name=cfg.dataset_name,
    #         device=device,
    #         model_path=Path(retrain_model_path),
    #     )

    #     retrain_models.append(retrain_model)

    # UPLOADING UNLEARNED MODEL ###
    print("Evaluating unlearned model accuracy...")
    unlearned_model, unlearned_retain_accuracy, unlearned_forget_accuracy = (
        evaluate_model_accuracy(
            model_path=Path(cfg.unlearned_model_path),
            cfg=cfg,
            retain_dataset=retain_dataset,
            forget_dataset=forget_dataset,
            class_names=class_names,
            mode="after_unlearning",
            device=device,
        )
    )
    # unlearned_model = get_cifar_resnet56(
    #     dataset_name=cfg.dataset_name,
    #     device=device,
    #     model_path=Path(cfg.unlearned_model_path),
    # )

    ### SIMILARITY SOURCE ###
    print("Calculating similarity...")
    class_mean_embeddings_retain, class_mean_embeddings_forget, similarity_xlabel = (
        compute_class_mean_embeddings(
            full_model, retain_dataset, forget_dataset, cfg, device=device
        )
    )

    ### SIMILARITY CALCULATION ###
    cosine_similarity_retain, cosine_similarity_forget = (
        compute_cosine_similarity_to_forget_class(
            class_mean_embeddings_retain=class_mean_embeddings_retain,
            class_mean_embeddings_forget=class_mean_embeddings_forget,
            class_to_forget=cfg.class_to_forget,
        )
    )

    _, retrain_retain_accuracy, retrain_forget_accuracy = evaluate_model_accuracy(
        model_path=Path(cfg.retrain_model_paths[0]),
        cfg=cfg,
        retain_dataset=retain_dataset,
        forget_dataset=forget_dataset,
        class_names=class_names,
        mode="retrain",
        device=device,
    )

    # Save raw arrays for later analysis
    save_accuracy_drop_vs_similarity_plot(
        retain_acc=retrain_retain_accuracy,
        forget_acc=retrain_forget_accuracy,
        unlearn_retain_acc=unlearned_retain_accuracy,
        unlearn_forget_acc=unlearned_forget_accuracy,
        cosine_similarity_retain=cosine_similarity_retain,
        cosine_similarity_forget=cosine_similarity_forget,
        forget_class=cfg.class_to_forget,
        class_names=class_names,
        output_path=cfg.output_dir / "accuracy_drop_vs_similarity.png",
        title=(
            f"Accuracy drop vs similarity to forget class "
            f"({cfg.dataset_name}, {class_names[cfg.class_to_forget]})"
        ),
        xlabel=similarity_xlabel,
        selected_classes=cfg.selected_classes,
    )

    print("Calculating extra metrics...")
    cfg.metrics = collect_extra_metrics(
        unlearned_model=unlearned_model,
        full_model=full_model,
        retain_dataset=retain_dataset,
        forget_dataset=forget_dataset,
        test_dataset=test_dataset,
        class_to_forget=cfg.class_to_forget,
        point_topk_fraction=cfg.point_topk_fraction,
        batch_size=cfg.unlearning_batch_size,
        num_workers=4,
        device=device,
        save_plots=True,
        plots_dir=cfg.output_dir,
    )

    if cfg.test_on_train:
        print("Calculating SVC-MIA forget efficacy...")
        cfg.svc_mia = compute_svc_mia_forget_efficacy(
            model=unlearned_model,
            retain_dataset=retain_dataset,
            forget_dataset=forget_dataset,
            test_dataset=test_dataset,
            batch_size=cfg.unlearning_batch_size,
            num_workers=4,
            device=device,
            seed=cfg.seed,
        )

    save_experiment_config(cfg, cfg.output_dir)

    print("Copying model config to the output folder")
    copy_unlearning_config_dir(
        Path(cfg.unlearned_model_path), cfg.output_dir / "unlearning_config"
    )
