from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def collect_unlearning_stats_globally(
    parent_dir: str | Path,
    method_names: list[str],
    class_to_forget: int,
    class_fraction_to_forget: float,
    output_json_path: str | Path,
) -> dict[str, Any]:
    parent_dir = Path(parent_dir)

    output_data: dict[str, Any] = {}

    for method_name in method_names:
        train_root = (
            parent_dir / method_name / f"class_{class_to_forget}" / "test_on_train=True"
        )
        test_root = (
            parent_dir
            / method_name
            / f"class_{class_to_forget}"
            / "test_on_train=False"
        )

        metric_lists = {
            "RA": [],
            "FA_forget_class": [],
            "RA_forget_class": [],
            "MIA": [],
            "TA": [],
            "TA_forget_class": [],
            "RTE": [],
        }
        used_runs = []

        if not train_root.exists() or not test_root.exists():
            output_data[method_name] = {
                "num_runs": 0,
                "used_ts_dirs": [],
                "metrics": {
                    key: {"values": [], "mean": None, "std": None}
                    for key in metric_lists
                },
            }
            continue

        train_ts_dirs = {p.name: p for p in train_root.iterdir() if p.is_dir()}
        test_ts_dirs = {p.name: p for p in test_root.iterdir() if p.is_dir()}

        ts_pairs = [
            (train_ts, test_ts)
            for train_ts, test_ts in zip(train_ts_dirs.keys(), test_ts_dirs.keys())
        ]

        for train_ts_dir_name, test_ts_dir_name in ts_pairs:
            train_ts_dir = train_ts_dirs[train_ts_dir_name]
            test_ts_dir = test_ts_dirs[test_ts_dir_name]

            train_config_path = train_ts_dir / "config.json"
            test_config_path = test_ts_dir / "config.json"
            rte_config_path = test_ts_dir / "unlearning_config" / "accuracies.json"

            if not (
                train_config_path.exists()
                and test_config_path.exists()
                and rte_config_path.exists()
            ):
                continue

            try:
                train_config = _load_json(train_config_path)
                test_config = _load_json(test_config_path)
                rte_config = _load_json(rte_config_path)
            except Exception:
                continue

            train_fraction = train_config.get("class_fraction_to_forget")
            test_fraction = test_config.get("class_fraction_to_forget")

            if (
                train_fraction != class_fraction_to_forget
                or test_fraction != class_fraction_to_forget
            ):
                continue

            try:
                stats = {
                    "RA": train_config["metrics"]["retain_accuracy"][
                        "after_unlearning"
                    ],
                    "FA_forget_class": train_config["metrics"]["forget_accuracy"][
                        "after_unlearning"
                    ],
                    "RA_forget_class": train_config["metrics"][
                        "retain_subset_true_label_equals_forget_class"
                    ]["unlearn"],
                    "MIA": train_config["svc_mia"]["unlearn"]["confidence"],
                    "TA": test_config["metrics"]["retain_accuracy"]["after_unlearning"],
                    "TA_forget_class": test_config["metrics"][
                        "retain_subset_true_label_equals_forget_class"
                    ]["unlearn"],
                    "RTE": rte_config["rte"],
                }
            except Exception:
                print(train_config_path)
                print(test_config_path)
                print(rte_config_path)

            for key, value in stats.items():
                metric_lists[key].append(float(value))

            used_runs.append((train_ts_dir_name, test_ts_dir_name))

        output_data[method_name] = {
            "num_runs": len(used_runs),
            "used_ts_dirs": used_runs,
            "metrics": {
                key: {
                    "values": values,
                    "mean": _mean(values),
                    "std": _std(values),
                }
                for key, values in metric_lists.items()
            },
        }

    _save_json(output_json_path, output_data)
    return output_data
