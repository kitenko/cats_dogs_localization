from typing import Tuple

import tensorflow as tf
import efficientnet.tfkeras as efn
from classification_models.tfkeras import Classifiers

from config import MODEL_NAME, INPUT_SHAPE, NUMBER_OF_CLASSES, WEIGHTS


def build_model(image_shape: Tuple[int, int, int] = INPUT_SHAPE, num_classes: int = NUMBER_OF_CLASSES,
                name_model: str = MODEL_NAME, weights: str = WEIGHTS) -> tf.keras.models.Model:
    """
    This function creates a model from 'classification_models.tf.keras' or 'efficientnet.tf.keras' library depending on
    the input parameters.

    :param image_shape: input shape (height, width, channels).
    :param num_classes: number of classes.
    :param name_model: the name of the model to be built.
    :param weights: ImageNet or None.
    :return: tf.keras.models.Model
    """
    models_dict = {
        'EfficientNetB0': efn.EfficientNetB0,
        'EfficientNetB1': efn.EfficientNetB1,
        'EfficientNetB2': efn.EfficientNetB2,
        'EfficientNetB3': efn.EfficientNetB3,
        'EfficientNetB4': efn.EfficientNetB4,
        'EfficientNetB5': efn.EfficientNetB5,
        'EfficientNetB6': efn.EfficientNetB6,
        'EfficientNetB7': efn.EfficientNetB7,
        'EfficientNetL2': efn.EfficientNetL2
    }

    try:
        if MODEL_NAME[:-2].lower() != 'efficientnet':
            name_model, preprocess_input = Classifiers.get(name_model)
            base_model = name_model(input_shape=image_shape, weights=weights, include_top=False)
        else:
            base_model = models_dict[name_model](input_shape=image_shape, weights=weights, include_top=False)
    except ValueError:
        print('the model name is incorrect, please check the model name')

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(5, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])

    return model
