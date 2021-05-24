from typing import Tuple
from abc import abstractmethod

import numpy as np

import tensorflow as tf

from config import INPUT_SHAPE
class Metric:
    def __init__(self, num_classes: int, is_binary_cross_entropy: bool = False) -> None:
        """
        # заменить текст
        Metrics are counted False Positive, False Negative, True Positive.

        :param num_classes: number of classes in the dataset.
        :param is_binary_cross_entropy: If there are no more than two classes, the value is set to True.
        """
        self.num_classes = num_classes
        self.epsilon = 1e-6
        self.is_binary_cross_entropy = is_binary_cross_entropy
        self.__name__ = 'metric'
        if self.is_binary_cross_entropy and self.num_classes != 2:
            msg = 'There should be 2 classes with binary cross entropy, got {}.'.format(self.num_classes)
            raise ValueError(msg)

    def confusion_matrix(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        This functions counts confusion_matrix.

        :param y_true: This is the true mark of data.
        :param y_pred: This is the predict mark of data.
        :return: False Positive, False Negative, True Positive.
        """
        if self.is_binary_cross_entropy:
            y_true = tf.cast(y_true > 0.5, tf.float32)[:, 0]
            y_pred = tf.cast(y_pred > 0.5, tf.float32)[:, 0]
        else:
            y_true = tf.argmax(y_true, 1)
            y_pred = tf.argmax(y_pred, 1)

        matrix = tf.cast(tf.math.confusion_matrix(y_true, y_pred, self.num_classes), tf.float32)
        fp = tf.reduce_sum(matrix, axis=0) - tf.linalg.tensor_diag_part(matrix)
        fn = tf.reduce_sum(matrix, axis=1) - tf.linalg.tensor_diag_part(matrix)
        tp = tf.linalg.tensor_diag_part(matrix)
        return fp, fn, tp

    @abstractmethod
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        raise NotImplementedError('This method must be implemented in subclasses')


class Recall(Metric):
    def __init__(self, num_classes: int, is_binary_cross_entropy: bool = False) -> None:
        """
        This class calculates the Recall metric.

        :param num_classes: number of classes in the dataset.
        :param is_binary_cross_entropy: If there are no more than two classes, the value is set to True.
        """
        super().__init__(num_classes, is_binary_cross_entropy)
        self.__name__ = 'recall'

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """

       :param y_true: This is the true mark of validation data.6
       :param y_pred: This is the predict mark of validation data.
       :return: recall recall metric.
       """
        fp, fn, tp = self.confusion_matrix(y_true, y_pred)
        return tp / (tp + fn + self.epsilon)


class Precision(Metric):
    def __init__(self, num_classes, is_binary_cross_entropy=False) -> None:
        """
        This class calculates the Precision metric.

        :param num_classes: number of classes in the dataset.
        :param is_binary_cross_entropy: If there are no more than two classes, the value is set to True.
        """
        super().__init__(num_classes, is_binary_cross_entropy)
        self.__name__ = 'precision'

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """

        :param y_true: This is the true mark of validation data.
        :param y_pred: This is the predict mark of validation data.6
        :return: precision metric.
        """
        fp, fn, tp = self.confusion_matrix(y_true, y_pred)
        return tp / (tp + fp + self.epsilon)


class F1Score(Metric):
    def __init__(self, num_classes: int, is_binary_cross_entropy: bool = False, beta: float = 1):
        """
        This class calculates the Precision metric.

        :param num_classes: number of classes in the dataset.
        :param is_binary_cross_entropy: If there are no more than two classes, the value is set to True.
        :param beta: with a coefficient of beta = 1, accuracy and completeness equally affect the value of the
                     F-measure, beta > 1 allows you to give more weight to completeness, and beta < 1-accuracy.
        """
        super().__init__(num_classes, is_binary_cross_entropy)
        self.beta = beta
        self.__name__ = 'F1_score'

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """

        :param y_true: This is the true mark of validation data.
        :param y_pred: This is the predict mark of validation data.6
        :return: f1score metric.
        """
        fp, fn, tp = self.confusion_matrix(y_true, y_pred)
        recall = tp / (tp + fn + self.epsilon)
        precision = tp / (tp + fp + self.epsilon)
        return (self.beta ** 2 + 1) * precision * recall / (self.beta ** 2 * precision + recall + self.epsilon)


class IoU(Metric):
    def __init__(self, num_classes: int, is_binary_cross_entropy: bool = False):
        super().__init__(num_classes, is_binary_cross_entropy)
        self.__name__ = 'IoU'

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:

        print(y_true, y_pred)
        # determine the (x, y)-coordinates of the intersection rectangle
        xTrue = tf.math.maximum(y_true[:, 1], y_pred[:, 1])
        yTrue = tf.math.maximum(y_true[:, 2], y_pred[:, 2])
        xPred = tf.math.minimum(y_true[:, 3], y_pred[:, 3])
        yPred = tf.math.minimum(y_true[:, 4], y_pred[:, 4])

        one = tf.ones(1)

        # compute the area of intersection rectangle
        interArea = tf.math.maximum(tf.zeros(1), xPred - xTrue + tf.ones(1)) * tf.math.maximum(tf.zeros(1), yPred - yTrue + tf.ones(1))

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (y_true[:, 3] - y_true[:, 1] + tf.ones(1)) * (y_true[:, 4] - y_true[:, 2] + tf.ones(1))
        boxBArea = (y_pred[:, 3] - y_pred[:, 1] + tf.ones(1)) * (y_pred[:, 4] - y_pred[:, 2] + tf.ones(1))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        # iou = tf.cast(interArea, tf.float32) / tf.cast((tf.cast(boxAArea, tf.float32) + tf.cast(boxBArea, tf.float32)
        #                                                 - tf.cast(interArea, tf.float32)), tf.float32)
        iou = interArea / (boxAArea + boxBArea - interArea)

        return iou
