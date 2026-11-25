"""Fine-tune a SwinUNETR or UNETR backbone on a subject split."""
import argparse

from hyps.config import load_config, select_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    select_gpu(cfg)
    from hyps.engine import train

    train(cfg)


if __name__ == "__main__":
    main()
