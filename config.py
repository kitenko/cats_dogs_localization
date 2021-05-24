import os
from datetime import datetime

DATASET_PATH = 'cats_dogs_dataset'
TRAIN_DATA = os.path.join(DATASET_PATH, 'train')
VAL_DATA = os.path.join(DATASET_PATH, 'valid')
JSON_FILE_PATH = os.path.join(DATASET_PATH, 'data.json')

BATCH_SIZE = 16
NUMBER_OF_CLASSES = 2
INPUT_SHAPE = (224, 224, 3)
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

date_time_for_save = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

LOGS_DIR_CURRENT_MODEL = os.path.join(LOGS, MODEL_NAME + '_' + str(WEIGHTS) + '_' + date_time_for_save)
SAVE_CURRENT_MODEL = os.path.join(SAVE_MODELS, MODEL_NAME + '_' + str(WEIGHTS) + '_' + date_time_for_save)
SAVE_CURRENT_TENSORBOARD_LOGS = os.path.join(TENSORBOARD_LOGS, MODEL_NAME + '_' + str(WEIGHTS) + '_' +
                                             date_time_for_save)
