import os
import argparse

import onnx
import keras2onnx

from src import build_model


def convert(path_keras_weights: str, path_save_onnx: str, onnx_model_name: str) -> None:
    """
    This function builds the keras model, loads weights and converts them to an onnx model.

    param path_keras_weights: The path to the model weights.
    param path_save_onnx: The path where the model will be saved.
    param onnx_model_name: Model name.
    """
    model = build_model()
    model.load_weights(path_keras_weights)
    onnx_model = keras2onnx.convert_keras(model, model.name)
    onnx.save_model(onnx_model, os.path.join(path_save_onnx, onnx_model_name))


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('script for model testing.')
    parser.add_argument('--weights', type=str, default=None, help='Path for loading model weights.')
    parser.add_argument('--save_onnx', type=str, default=None, help='Path for save onnx model.')
    parser.add_argument('--name_model', type=str, default='onnx_model.onnx', help='Name model.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert(path_keras_weights=args.weights, path_save_onnx=args.save_onnx, onnx_model_name=args.name_model)
