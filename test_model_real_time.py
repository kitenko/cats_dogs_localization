import os
from typing import Tuple
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A

from model import build_model
from config import INPUT_SHAPE


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


def image_vis():
    # When everything done, release the capture
    cat_dog = {0: 'cat', 1: 'dog'}
    BOX_COLOR = (255, 0, 0)  # Red
    TEXT_COLOR = (255, 255, 255)  # White

    model = build_model()
    model.load_weights('models_data/save_models/resnet18_imagenet_2021-05-23 18:51:27/resnet18.h5')
    path = '/home/andre/Downloads/1560872802_0_778_1536_1642_600x0_80_0_0_606c2d47b6d37951adc9eaf750de22f0.jpg'
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
    predict = model.predict(np.expand_dims(img, axis=0) / 255.0)[0]
    if predict[0] <= 0.5:
        label = 0
    else:
        label = 1
    aug = A.Compose([A.Resize(height=720, width=720)],
                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    transform = aug(image=img, bboxes=[predict[1:] * INPUT_SHAPE[0]], category_ids=[str(predict[0])])
    img = transform['image']
    bounding_box = transform['bboxes'][0]
    x_min, y_min, x_max, y_max = (int(bounding_box[0]),
                                  int(bounding_box[1]),
                                  int(bounding_box[2]),
                                  int(bounding_box[3]))
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=BOX_COLOR, thickness=2)
    ((text_width, text_height), _) = cv2.getTextSize(cat_dog[label],
                                                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=cat_dog[label],
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(devices[0], True)
    visualization()
    #image_vis()

