from typing import Literal

DatasetName = Literal["cifar10", "cifar100", "svhn"]
UnlearningMethod = Literal[
    "random_labels",
    "negative_grad",
    "random_labels_orig",
    "salun",
    "trw",
    "amun",
    "distill_labels",
]
SimilaritySource = Literal["imagenet_resnet18", "full_model"]
GradientLayerScope = Literal["last_layer", "last_block_and_head"]
RandomLabelMode = Literal["wrong", "any"]
RetainSelectionMode = Literal["random", "closest", "farthest"]
