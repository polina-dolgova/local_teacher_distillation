# Local Teacher Distillation

This repository contains the code accompanying our submission.
It implements experimental pipelines for studying **machine unlearning** across multiple settings, including large-scale image classification, small-scale benchmarks, and controlled synthetic experiments.

---
## Repository Structure

The codebase is organized by experimental setting:

* **`cifar/` — Image Classification (CIFAR-100)**
  Large-scale experiments with deep neural networks (ResNet-56).
  Includes implementations of multiple unlearning methods, retraining baselines, and evaluation pipelines.

* **`mnist/` — Small-scale Experiments**
  Controlled experiments on MNIST for additional validation and analysis.

* **`regression/` — Synthetic Experiments**
  Experiments in a simplified regression setting, used to study theoretical properties and isolate specific effects.

Detailed instructions for running experiments are provided in `cifar/README.md`.

---

## Experimental Pipeline

Across all settings, we follow a unified pipeline:

1. **Training / Retraining**
   Train a model either on the full dataset or on the retained subset.

2. **Unlearning**
   Apply an unlearning method to remove the effect of selected data (e.g., a class or subset).

3. **Evaluation**
   Measure performance on:

   * retained data
   * forgotten data
   * test data

---

## Reproducibility

* Each run saves:

  * model checkpoints (`.pth`)
  * configuration files (`config.json`)
  * evaluation metrics (`accuracies.json`)
* Experiments are fully configurable via command-line arguments.
* Random seeds and hyperparameters are explicitly controlled via configuration files.
