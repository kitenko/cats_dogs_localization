import os
import json
from typing import Dict

from tensorflow import keras

from config import SAVE_MODEL_EVERY_EPOCH


class LogCallback(keras.callbacks.Callback):
    def __init__(self, model_save_path: str, logs_save_path: str, save_model_every_era: bool =
                 SAVE_MODEL_EVERY_EPOCH) -> None:
        """
        Logging all training metrics to a json file and saving the model every epoch.

        :param model_save_path: path to the folder in which to save the models.
        :param logs_save_path: path to the folder in which to save the logs.
        :param save_model_every_era: save the model or not at the end of each epoch.
        """
        super().__init__()
        self.model_save_path = model_save_path
        self.log_file = os.path.join(logs_save_path, 'train_logs.json')
        self.logs = {'epochs': []}
        self.save_model_every_epoch = save_model_every_era

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_logs()
        if self.save_model_every_epoch:
            self.model.save(os.path.join(self.model_save_path, 'last.h5'))

    def save_logs(self) -> None:
        with open(self.log_file, 'w') as file:
            json.dump(self.logs, file, indent=4)

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None) -> None:
        """
        This method prepares the data for writing to a json file and calls the method to save the json file.
        If  = self.save_model_every_epoch = True, then the model is saved every epoch.

        :param epoch: current epoch number.
        :param logs: dictionary with metrics that are specified in the model, such as accuracy, precision or recall.
        """
        text = ['epoch: {:03d}'.format(epoch + 1)]
        for key, value in logs.items():
            if key not in self.logs:
                self.logs[key] = []
            self.logs[key].append(float(value))
            text.append('{}: {:.04f}'.format(key, float(value)))
        self.logs['epochs'].append('; '.join(text))
        self.save_logs()
        if self.save_model_every_epoch:
            self.model.save(os.path.join(self.model_save_path, '{:03d}_epoch.h5'.format(1 + epoch)))
