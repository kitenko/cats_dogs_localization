import time
import argparse
from tqdm import tqdm
from typing import Tuple, List

import cv2
import numpy as np
import tensorflow as tf

from src import build_model
from src import DataGenerator
from config import INPUT_SHAPE
from src import Accuracy, IoURectangle


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('script for model testing.')
    parser.add_argument('--weights', type=str, default=None, help='Path for loading model weights.')
    parser.add_argument('--webcam', action='store_true', help='If the value is True, then the webcam will be used for '
                                                              'the test.')
    parser.add_argument('--metrics', action='store_true', help='If the value is True, then the average metrics on the '
                                                               'validation dataset will be calculated.')
    parser.add_argument('--time', action='store_true', help='If the value is True, then the inference time and the '
                                                            'average fps on the validation dataset will be calculated.')
    parser.add_argument('--gpu', action='store_true', help='If True, then the gpu is used for the test.')
    return parser.parse_args()


def preparing_frame(image: np.ndarray, model: tf.keras.Model, cam_width: float, cam_height: float) -> Tuple[List[int],
                                                                                                            int]:
    """
    This function prepares the image and makes a prediction.

    :param cam_height:
    :param cam_width:
    :param image: this is input image or frame.
    :param model: model with loaded weights.
    :return: bounding_box and class for image.
    """
    image = cv2.resize(image, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
    predict = model.predict(np.expand_dims(image, axis=0) / 255.0)[0]
    if predict[0] <= 0.5:
        label = 0
    else:
        label = 1

    bounding_box = [int(predict[1]*cam_width), int(predict[2]*cam_height),  int(predict[3]*cam_width),
                    int(predict[4]*cam_height)]

    return bounding_box, label


def visualization() -> None:
    """
    This function captures webcam video and resizes the image.
    """
    cat_dog = {0: 'cat', 1: 'dog'}
    BOX_COLOR = (255, 0, 0)  # Red
    TEXT_COLOR = (255, 255, 255)  # White

    model = build_model()
    model.load_weights(args.weights)

    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    prev_frame_time = 0
    while True:
        ret, frame = cap.read()

        bounding_box, label = preparing_frame(image=frame, model=model, cam_width=width, cam_height=height)

        x_min, y_min, x_max, y_max = bounding_box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=BOX_COLOR, thickness=2)
        ((text_width, text_height), _) = cv2.getTextSize(cat_dog[label],
                                                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(frame, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
        cv2.putText(
            frame,
            text=cat_dog[label],
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, str(int(fps)) + ':fps', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def test_metrics_and_time(mode: str) -> None:
    """
    This function calculates the average value of loss and metrics as well as inference time and average fps.

    :param mode: depending on the mode ('metrics', 'time'), the function counts (loss, metrics) or time and average fps.
    """
    data_gen = DataGenerator(batch_size=1, is_train=False)
    model = build_model()
    model.load_weights(args.weights)
    model.compile(loss=tf.keras.losses.binary_crossentropy, metrics=[Accuracy(), IoURectangle()])

    if mode == 'metrics':
        metrics = model.evaluate(data_gen, workers=8)
        print('Loss:{:.2f}'.format(metrics[0]), 'Accuracy:{:.2f}%'.format(metrics[1]),
              'IoURectangle:{:.2f}%'.format(metrics[2]))

    elif mode == 'time':
        all_times = []
        for i in tqdm(range(len(data_gen))):
            images, _ = data_gen[i]
            start_time = time.time()
            model.predict(images)
            finish_time = time.time()
            all_times.append(finish_time - start_time)
        all_times = all_times[5:]
        message = '\nMean inference time: {:.04f}. Mean FPS: {:.04f}.\n'.format(
            np.mean(all_times),
            len(all_times) / sum(all_times))
        print(message)


if __name__ == '__main__':
    args = parse_args()

    if args.gpu is True:
        devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(devices[0], True)

    if args.webcam is True:
        visualization()
    if args.metrics is True:
        test_metrics_and_time('metrics')
    if args.time is True:
        test_metrics_and_time('time')
