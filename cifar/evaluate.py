from cifar.src.config import parse_config, save_experiment_config
from cifar.src.utils import get_default_device, build_output_dir
from cifar.src.pipeline import run_experiment


def main(cfg, output_dir, device):
    save_experiment_config(cfg, output_dir)

    run_experiment(cfg, device=device)


if __name__ == "__main__":
    cfg = parse_config()
    device = get_default_device()
    output_dir = build_output_dir(cfg)
    cfg.output_dir = output_dir  # Update config with resolved output dir

    main(cfg=cfg, output_dir=output_dir, device=device)
