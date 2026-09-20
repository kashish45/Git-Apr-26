import argparse
import pickle
from pathlib import Path

import numpy as np


def load_model(model_path):
    model_path = Path(model_path)
    with model_path.open("rb") as f:
        model = pickle.load(f)
    return model


def predict(model, features):
    features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    return model.predict(features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a pickled ML model and run prediction.")
    parser.add_argument("--model_path", type=str, default="model.pkl", help="Path to the pickled model file.")
    parser.add_argument(
        "--features",
        type=float,
        nargs="+",
        required=True,
        help="Feature values for prediction. Example: --features 5.1 3.5 1.4 0.2",
    )
    args = parser.parse_args()

    model = load_model(args.model_path)
    predictions = predict(model, args.features)
    print(predictions)
