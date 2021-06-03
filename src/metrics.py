import tensorflow as tf


class IoURectangle(tf.keras.metrics.Metric):
    def __init__(self, name: str = 'iou_rectangle', **kwargs) -> None:
        """
        This class is first initialized in variables with zeros, then calculates the batch metric 'iou_rectangle' and
        adds the thread to the total value using 'assign_add'. After the number of resulting metrics is divided by the
        number of batch, then at the end of the epoch, the results are reset to zero.

        :param name: this name metric.
        """
        super(IoURectangle, self).__init__(name=name, **kwargs)
        self.ious = self.add_weight(name='iou_rectangle', initializer='zeros')
        self.batches = self.add_weight(name='batch', initializer='zeros')

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> None:
        """
        This method computes the area of intersection rectangle, compute the area of both the prediction and
        ground-truth rectangles and computes the intersection over union by taking the intersection area and
        dividing it by the sum of prediction + ground-truth areas - the intersection area.

        :param y_true: This is the true mark of data.
        :param y_pred: This is the predict mark of data.
        :param sample_weight:
        """
        x_max = tf.math.maximum(y_true[:, 1], y_pred[:, 1])
        y_max = tf.math.maximum(y_true[:, 2], y_pred[:, 2])
        x_min = tf.math.minimum(y_true[:, 3], y_pred[:, 3])
        y_min = tf.math.minimum(y_true[:, 4], y_pred[:, 4])

        inter_area = (tf.math.maximum(tf.zeros(1), x_min - x_max + tf.ones(1)) *
                      tf.math.maximum(tf.zeros(1), y_min - y_max + tf.ones(1)))

        first_area = (y_true[:, 3] - y_true[:, 1] + tf.ones(1)) * (y_true[:, 4] - y_true[:, 2] + tf.ones(1))
        second_area = (y_pred[:, 3] - y_pred[:, 1] + tf.ones(1)) * (y_pred[:, 4] - y_pred[:, 2] + tf.ones(1))

        iou = inter_area / (first_area + second_area - inter_area)
        self.ious.assign_add(tf.reduce_sum(iou))
        self.batches.assign_add(tf.cast(tf.shape(iou)[0], tf.float32))

    def result(self):
        return self.ious / self.batches

    def reset_state(self):
        self.ious.assign(0)
        self.batches.assign(0)


class Accuracy(tf.keras.metrics.BinaryAccuracy):
    def __init__(self, name='accuracy_custom', **kwargs):
        """
        This class inherits from tf. keras.metrics. Binary accuracy, the "accuracy" metric is calculated only for the
        cat or dog class.

        :param name: this name metric.
        """
        super(Accuracy, self).__init__(name=name, **kwargs)

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        """
        This method computes accuracy only for class.

        :param y_true: This is the true mark of data.
        :param y_pred: This is the predict mark of data.
        :param sample_weight:
        """
        y_true = tf.expand_dims(y_true[:, 0], axis=1)
        y_pred = tf.expand_dims(y_pred[:, 0], axis=1)
        return super(Accuracy, self).update_state(y_true, y_pred, sample_weight)
