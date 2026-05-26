import argparse
from pathlib import Path

from scripts.inference_stageA import run_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Stage A inference")

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    run_inference(
        input_path=Path(args.input),
        checkpoint_path=Path(args.checkpoint),
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()