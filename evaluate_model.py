import sys
import os
from pathlib import Path

from modules.lab_utils import load_dataset, LAB_PATH, fix_path


def evaluate_pytorch_model(dataset_path, model_path):
    from modules.pytorch_mlp import PytorchMLPReg

    mlp = PytorchMLPReg(model_file=model_path)
    _, _, x_test, y_test = load_dataset(dataset_path)

    score = mlp.score(x_test, y_test)
    print(f"Score on dataset: {score:.4e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on a dataset"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["pytorch", "calibrated"],
        default="pytorch",
        help="Model type: pytorch or calibrated",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=False,
        default=Path("data/results/blueleg_beam_sphere515.csv"),
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=False,
        default=Path("data/results/blueleg_beam_sphere.pth"),
        help="Path to trained model file",
    )

    args = parser.parse_args()

    model_type = args.model_type
    dataset_path = args.dataset_path
    model_path = args.model_path

    dataset_path = fix_path(dataset_path)
    if dataset_path is None:
        print(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    model_path = fix_path(model_path)
    if model_path is None:
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    if model_type == "pytorch":
        evaluate_pytorch_model(str(dataset_path), str(model_path))
    elif model_type == "calibrated":
        print("Calibrated model evaluation is not implemented yet.")
        sys.exit(2)
    else:
        print(f"Unknown model type: {model_type}")
        sys.exit(1)
