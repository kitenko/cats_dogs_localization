import os
from datetime import datetime

DATASET_PATH = 'cats_dogs_dataset'
TRAIN_DATA = os.path.join(DATASET_PATH, 'train')
VAL_DATA = os.path.join(DATASET_PATH, 'valid')

BATCH_SIZE = 8
NUMBER_OF_CLASSES = 2
INPUT_SHAPE = (256, 256, 3)
LEARNING_RATE = 0.0001
EPOCHS = 150
WEIGHTS = 'imagenet'
USE_AUGMENTATION = False
SAVE_MODEL_EVERY_EPOCH = False

MODEL_NAME = 'resnet18'

MODELS_DATA = 'models_data'
TENSORBOARD_LOGS = os.path.join(MODELS_DATA, 'tensorboard_logs')
SAVE_MODELS = os.path.join(MODELS_DATA, 'save_models')
LOGS = os.path.join(MODELS_DATA, 'logs')

date_time_for_save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

LOGS_DIR_CURRENT_MODEL = os.path.join(LOGS, MODEL_NAME + '_' + str(WEIGHTS) + '_' + date_time_for_save + '_' +
                                      'AUG_' + str(USE_AUGMENTATION))
SAVE_CURRENT_MODEL = os.path.join(SAVE_MODELS, MODEL_NAME + '_' + str(WEIGHTS) + '_' + date_time_for_save + '_' +
                                  'AUG_' + str(USE_AUGMENTATION))
SAVE_CURRENT_TENSORBOARD_LOGS = os.path.join(TENSORBOARD_LOGS, MODEL_NAME + '_' + str(WEIGHTS) + '_' +
                                             date_time_for_save + '_' + 'AUG_' + str(USE_AUGMENTATION))
