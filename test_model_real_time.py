import os
import time
import argparse
from typing import Tuple
from tqdm import tqdm

import cv2
import numpy as np
import tensorflow as tf
import albumentations as A

from model import build_model
from data_generator import DataGenerator
from config import INPUT_SHAPE
from metrics import Accuracy, IoURectangle


def preparing_frame(image: np.ndarray, model) -> Tuple[np.ndarray, tuple, int]:
    """
    This function prepares the image and makes a prediction.

    :param image: this is input image or frame.
    :param model: model with loaded weights.
    :return:
    """
    image = cv2.resize(image, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
    predict = model.predict(np.expand_dims(image, axis=0) / 255.0)[0]
    if predict[0] <= 0.5:
        label = 0
    else:
        label = 1

    aug = A.Compose([A.Resize(height=720, width=720)],
                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    transform = aug(image=image, bboxes=[predict[1:] * INPUT_SHAPE[0]], category_ids=[str(predict[0])])
    frame = transform['image']
    bounding_box = transform['bboxes'][0]

    return frame, bounding_box, label


def visualization() -> None:
    """
    This function captures webcam video and resizes the image.
    """
    cat_dog = {0: 'cat', 1: 'dog'}
    BOX_COLOR = (255, 0, 0)  # Red
    TEXT_COLOR = (255, 255, 255)  # White

    model = build_model()
    model.load_weights('models_data/save_models/resnet18_imagenet_2021-05-23 18:51:27/resnet18.h5')

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        predict = preparing_frame(image=frame, model=model)

        x_min, y_min, x_max, y_max = (int(predict[1][0]),
                                      int(predict[1][1]),
                                      int(predict[1][2]),
                                      int(predict[1][3]))
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=BOX_COLOR, thickness=2)
        ((text_width, text_height), _) = cv2.getTextSize(cat_dog[predict[2]],
                                                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(frame, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
        cv2.putText(
            frame,
            text=cat_dog[predict[2]],
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )

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
    model.load_weights('models_data/save_models/resnet18_imagenet_2021-05-23 18:51:27/resnet18.h5')
    model.compile(loss=tf.keras.losses.binary_crossentropy, metrics=[Accuracy(), IoURectangle()])

    if mode == 'metrics':
        print(model.evaluate(data_gen, workers=8))

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
            len(all_times) / sum(all_times)
        )
        print(message)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(devices[0], True)
    # visualization()
    test_metrics_and_time('time')
