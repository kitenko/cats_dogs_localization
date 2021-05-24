import json
import argparse

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('script for plotting graphs.')
    parser.add_argument('--logs', type=str, default=None, help='Path for loading json file with logs.')
    return parser.parse_args()


def plot_graphics(path_for_json_file_logs: str, graf_loss: bool = True,
                  graf_accuracy: bool = True, graf_recall: bool = True, graf_precision: bool = True,
                  graf_f1score: bool = True) -> None:
    """
    Depending on the entered parameters, the function will plot different graphs.

    :param path_for_json_file_logs: This is the path for the json file with logs.
    :param graf_loss: To plot or not to plot the loss.
    :param graf_accuracy: To plot or not to plot the accuracy.
    :param graf_recall: To plot or not to plot the recall.
    :param graf_precision: To plot or not to plot the precision.
    :param graf_f1score: To plot or not to plot the f1score.
    """

    with open(path_for_json_file_logs) as json_file:
        data = json.load(json_file)

        # parameters for plots
        acc = data['accuracy']
        val_acc = data['val_accuracy']
        loss = data['loss']
        val_loss = data['val_loss']
        precision = data['precision']
        val_precision = data['val_precision']
        recall = data['recall']
        val_recall = data['val_recall']
        f1_score = data['F1_score']
        val_f1_score = data['val_F1_score']
        epochs_range = range(len(data['epochs']))

        plt.figure(figsize=(20, 20))

        if graf_accuracy:
            plt.subplot(1, 5, 1)
            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')

        if graf_loss:
            plt.subplot(1, 5, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')

        if graf_precision:
            plt.subplot(1, 5, 3)
            plt.plot(epochs_range, precision, label='Training Precision')
            plt.plot(epochs_range, val_precision, label='Validation Precision')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Precision')

        if graf_recall:
            plt.subplot(1, 5, 4)
            plt.plot(epochs_range, recall, label='Training Recall')
            plt.plot(epochs_range, val_recall, label='Validation Recall')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Recall')

        if graf_f1score:
            plt.subplot(1, 5, 5)
            plt.plot(epochs_range, f1_score, label='Training F1_Score')
            plt.plot(epochs_range, val_f1_score, label='Validation F1_Score')
            plt.legend(loc='upper right')
            plt.title('Training and Validation F1_Score')

        plt.show()


if __name__ == '__main__':
    plot_graphics(path_for_json_file_logs=parse_args().logs)
