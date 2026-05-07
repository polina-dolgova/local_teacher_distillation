# CIFAR Unlearning Experiments

This repository contains implementations of retraining and several unlearning methods for CIFAR-100 with a ResNet-56 backbone.

---

## ⭐ Our Method (Distillation-based)

```bash
python -m cifar.methods.distill.distill_labels \
  --model-path <MODEL_PATH> \
  --dataset-name <DATASET_NAME> \
  --class-to-forget <CLASS_TO_FORGET> \
  --class-fraction-to-forget <FRACTION> \
  --mode <MODE> \
  --teacher-type <TEACHER_TYPE> \
  --support-epochs <SUPPORT_EPOCHS> \
  --n-support <N_SUPPORT> \
  --num-epochs <NUM_EPOCHS> \
  --retain-loss-weight <RETAIN_WEIGHT> \
  --forget-loss-weight <FORGET_WEIGHT> \
  --unlearning-lr <LR> \
  --teacher-accuracy-threshold <THRESHOLD> \
  --support-lr <SUPPORT_LR> \
  --softlabels-target-mode <TARGET_MODE> \
  --save-dir <OUTPUT_DIR> \
  --seed <SEED>
```

---

## 📊 Evaluation

### Test set

```bash
python -m cifar.evaluate \
  --unlearned-model-path <MODEL_PATH> \
  --retrain-model-path <RETRAIN_MODEL_PATH> \
  --save-dir <OUTPUT_DIR> \
  --unlearning-method <METHOD_NAME> \
  --similarity-source full_model \
  --class-to-forget <CLASS_TO_FORGET> \
  --class-fraction-to-forget <FRACTION>
```

### Train set

```bash
python -m cifar.evaluate \
  --unlearned-model-path <MODEL_PATH> \
  --retrain-model-path <RETRAIN_MODEL_PATH> \
  --save-dir <OUTPUT_DIR> \
  --unlearning-method <METHOD_NAME> \
  --similarity-source full_model \
  --class-to-forget <CLASS_TO_FORGET> \
  --class-fraction-to-forget <FRACTION> \
  --test-on-train
```

---

## 🔁 Retraining

```bash
python -m cifar.retrain.train_resnet56_filtered \
  --save-dir <OUTPUT_DIR> \
  --dataset-name <DATASET_NAME> \
  --class-to-forget <CLASS_TO_FORGET> \
  --class-fraction-to-forget <FRACTION> \
  --nesterov \
  --device <DEVICE> \
  --seed <SEED>
```

---

## 🧠 Unlearning Methods

All methods require a fully trained model via `--model-path`.

---

## Implementation Notes

The SalUn and Fisher/WoodFisher (IU) implementations are based on the public SalUn codebase. The AMUN implementation is based on the official AMUN codebase.

---

### Gradient Ascent (GA)

```bash
python -m cifar.methods.ga \
  --model-path <MODEL_PATH> \
  --dataset-name <DATASET_NAME> \
  --class-to-forget <CLASS_TO_FORGET> \
  --class-fraction-to-forget <FRACTION> \
  --num-epochs <NUM_EPOCHS> \
  --unlearning-lr <LR> \
  --save-dir <OUTPUT_DIR> \
  --seed <SEED>
```

---

### Random Labels (RL)

```bash
python -m cifar.methods.rl \
  --model-path <MODEL_PATH> \
  --dataset-name <DATASET_NAME> \
  --class-to-forget <CLASS_TO_FORGET> \
  --class-fraction-to-forget <FRACTION> \
  --num-epochs <NUM_EPOCHS> \
  --unlearning-lr <LR> \
  --save-dir <OUTPUT_DIR> \
  --seed <SEED>
```

---

### Fine-Tuning (FT)

```bash
python -m cifar.methods.ft \
  --model-path <MODEL_PATH> \
  --dataset-name <DATASET_NAME> \
  --class-to-forget <CLASS_TO_FORGET> \
  --class-fraction-to-forget <FRACTION> \
  --num-epochs <NUM_EPOCHS> \
  --unlearning-lr <LR> \
  --save-dir <OUTPUT_DIR> \
  --seed <SEED>
```

---

### SALUN

```bash
python -m cifar.methods.salun \
  --model-path <MODEL_PATH> \
  --dataset-name <DATASET_NAME> \
  --class-to-forget <CLASS_TO_FORGET> \
  --class-fraction-to-forget <FRACTION> \
  --num-epochs <NUM_EPOCHS> \
  --unlearning-lr <LR> \
  --mask-topk-ratio <MASK_RATIO> \
  --save-dir <OUTPUT_DIR> \
  --seed <SEED>
```

---

### AMUN

```bash
python -m cifar.methods.amun \
  --model-path <MODEL_PATH> \
  --dataset-name <DATASET_NAME> \
  --class-to-forget <CLASS_TO_FORGET> \
  --class-fraction-to-forget <FRACTION> \
  --num-epochs <NUM_EPOCHS> \
  --unlearning-lr <LR> \
  --unlearn-method advonly \
  --save-dir <OUTPUT_DIR> \
  --seed <SEED>
```

---

### Fisher (WoodFisher)

```bash
python -m cifar.methods.fisher \
  --model-path <MODEL_PATH> \
  --dataset-name <DATASET_NAME> \
  --class-to-forget <CLASS_TO_FORGET> \
  --class-fraction-to-forget <FRACTION> \
  --alpha <ALPHA> \
  --save-dir <OUTPUT_DIR> \
  --seed <SEED>
```

---

## 📁 Outputs

Each run produces:

* `unlearned_model.pth`
* `config.json`
* `accuracies.json`
