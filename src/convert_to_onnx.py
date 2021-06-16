import os
import argparse

import onnx
import keras2onnx

from src import build_model


def convert(path_keras_weights: str, path_save_onnx: str, onnx_model_name: str):

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
    parser.add_argument('--onnx_model', type=str, default=None, help='Path for save onnx model.')
    parser.add_argument('--name_model', type=str, default='onnx_model', help='Name model.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert(path_keras_weights=args.weights, path_save_onnx=args.onnx_model, onnx_model_name=args.name_model)
