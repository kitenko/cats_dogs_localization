import os
import argparse

import cv2
import numpy as np
import albumentations as A

from model import build_model
from config import INPUT_SHAPE


def image_vis(weights: str, path_image: str) -> None:
    """
    This function makes a prediction for a single image and outputs it.

    param weights: this is path for weights.
    param path_image: this is path for model.
    """
    cat_dog = {0: 'cat', 1: 'dog'}
    box_color = (255, 0, 0)
    text_color = (255, 255, 255)

    model = build_model()
    model.load_weights(weights)
    path = path_image
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
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=box_color, thickness=2)
    ((text_width, text_height), _) = cv2.getTextSize(cat_dog[label],
                                                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), box_color, -1)
    cv2.putText(
        img,
        text=cat_dog[label],
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=text_color,
        lineType=cv2.LINE_AA,
    )
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('script for model testing.')
    parser.add_argument('--weights', type=str, default=None, help='Path for loading model weights.')
    parser.add_argument('--image_path', type=str, default=None, help='Path for loading model weights.')
    return parser.parse_args()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    args = parse_args()
    # devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(devices[0], True)
    image_vis(weights=args.weights, path_image=args.image_path)
