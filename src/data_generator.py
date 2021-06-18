import os
from typing import Tuple

import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from tensorflow import keras

from config import NUMBER_OF_CLASSES, BATCH_SIZE, INPUT_SHAPE, AUGMENTATION_DATA


class DataGenerator(keras.utils.Sequence):
    def __init__(self, train_data: str, val_data: str, batch_size: int = BATCH_SIZE, is_train: bool = True,
                 image_shape: Tuple[int, int, int] = INPUT_SHAPE, num_classes: int = NUMBER_OF_CLASSES,
                 augmentation_data: bool = AUGMENTATION_DATA) -> None:
        """
        Data generator for the task of colour images classifying.

        :param batch_size: number of images in one batch.
        :param is_train: if is_train = True, then we work with train images, otherwise with test.
        :param image_shape: this is image shape (height, width, channels).
        :param num_classes: number of image classes.
        :param augmentation_data: if this parameter is True, then augmentation is applied to the training dataset.
        """
        self.batch_size = batch_size
        self.is_train = is_train
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.augmentation_data = augmentation_data
        self.train_data = train_data
        self.val_data = val_data

        if is_train:
            images = [i for i in os.listdir(train_data) if i.endswith('.jpg')]
            augmentation = images_augmentation(use_augmentation=self.augmentation_data)
            self.path_data = train_data
        else:
            images = [i for i in os.listdir(train_data) if i.endswith('.jpg')]
            augmentation = images_augmentation(use_augmentation=False)
            self.path_data = val_data

        self.aug = augmentation
        self.data = images
        self.on_epoch_end()

    def on_epoch_end(self) -> None:
        """
        Random shuffling of data at the end of each epoch.

        """
        np.random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data) // self.batch_size + (len(self.data) % self.batch_size != 0)

    def __getitem__(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function makes batch.

        :param batch_idx: batch number.
        :return: image tensor and list with labels tensors for each output.
        """
        batch = self.data[(batch_idx * self.batch_size):((batch_idx + 1) * self.batch_size)]
        images = np.zeros((self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        labels = np.zeros((self.batch_size, 5))
        for i, img in enumerate(batch):
            name_txt_file = os.path.splitext(img)[0] + '.txt'
            img = cv2.imread(os.path.join(self.path_data, img))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with open(os.path.join(self.path_data, name_txt_file)) as f:
                f = f.readline()
                f = [int(n) for n in f.split()]
            aug = self.aug(image=img, bboxes=[f[1:]], category_ids=[str(f[0]-1)])
            img = aug['image']
            images[i, :, :, :] = img
            bounding_box = aug['bboxes'][0]
            labels[i, 0] = int(aug['category_ids'][0])
            labels[i, 1] = float(bounding_box[0]/self.image_shape[0])
            labels[i, 2] = float(bounding_box[1]/self.image_shape[1])
            labels[i, 3] = float(bounding_box[2]/self.image_shape[0])
            labels[i, 4] = float(bounding_box[3]/self.image_shape[1])
        images = image_normalization(images)
        return images, labels

    def show(self) -> None:
        """
        This method showing image with label.
        """
        box_color = (255, 0, 0)
        text_color = (255, 255, 255)

        for i in range(len(self)):

            images, labels = self[i]
            cat_dog = {0: 'cat', 1: 'dog'}
            rows_columns_subplot = self.batch_size

            while np.math.sqrt(rows_columns_subplot) - int(np.math.sqrt(rows_columns_subplot)) != 0.0:
                rows_columns_subplot += 1
            rows_columns_subplot = int(np.math.sqrt(rows_columns_subplot))
            plt.figure(figsize=[20, 20])
            for i, j in enumerate(images):
                x_min, y_min, x_max, y_max = (int(self.image_shape[0] * labels[i, 1]),
                                              int(self.image_shape[0] * labels[i, 2]),
                                              int(self.image_shape[0] * labels[i, 3]),
                                              int(self.image_shape[0] * labels[i, 4]))
                cv2.rectangle(j, (x_min, y_min), (x_max, y_max), color=box_color, thickness=2)
                ((text_width, text_height), _) = cv2.getTextSize(cat_dog[int(labels[i][0])],
                                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.rectangle(j, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), box_color, -1)
                cv2.putText(
                    j,
                    text=cat_dog[labels[i][0]],
                    org=(x_min, y_min - int(0.3 * text_height)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.35,
                    color=text_color,
                    lineType=cv2.LINE_AA,
                )
                plt.subplot(rows_columns_subplot, rows_columns_subplot, i+1)
                plt.imshow(j)
                if self.train_data is True:
                    plt.title('Augmented, class = "{}"'.format(cat_dog[labels[i][0]]))
                else:
                    plt.title('Original, class = "{}"'.format(cat_dog[labels[i][0]]))
            if plt.waitforbuttonpress(0):
                plt.close('all')
                raise SystemExit
            plt.close()


def images_augmentation(use_augmentation: bool = AUGMENTATION_DATA) -> A.Compose:
    """
    This function makes augmentation data.

    :param use_augmentation: if this parameter is True then augmentation is applied to train dataset.
    :return: augment data
    """
    if use_augmentation is True:
        aug = A.Compose([
              A.Resize(height=INPUT_SHAPE[0], width=INPUT_SHAPE[1]),
              A.Blur(blur_limit=(1, 4), p=0.2),
              A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.2),
              A.ColorJitter(brightness=0.1, contrast=0.0, saturation=0.1, hue=0.0, p=0.2),
              A.Equalize(mode='cv', by_channels=True, mask=None, p=0.1),
              A.Rotate(limit=50, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False,
                       p=0.2)
              ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    else:
        aug = A.Compose([A.Resize(height=INPUT_SHAPE[0], width=INPUT_SHAPE[1])],
                        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    return aug


def image_normalization(image: np.ndarray) -> np.ndarray:
    """
    Image normalization.
    :param image: image numpy array.
    :return: normalized image.
    """
    return image / 255.0


if __name__ == '__main__':
    x = DataGenerator(train_data=os.path.join('cats_dogs_dataset', 'train'),
                      val_data=os.path.join('cats_dogs_dataset', 'valid'))
    x.show()
