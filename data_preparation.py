import os
import json

from config import JSON_FILE_PATH, DATASET_PATH, VAL_DATA, TRAIN_DATA


def make_data_json(path_for_json: str = JSON_FILE_PATH, data_image: str = DATASET_PATH,
                   proportion_test_images: float = 0.2) -> None:
    """
    This function creates json file with train and test data of images.

    :param path_for_json: this is path where file will save.
    :param proportion_test_images: percentage of test images.
    :param data_image: path for data of images
    """

    path_images, name_images, label_images, xmin, ymin, xmax, ymax = [], [], [], [], [], [], []

    for i in enumerate(os.listdir(TRAIN_DATA)):
        try:
            if file.endswith('.jpg'):
                path_images.append(os.path.join(TRAIN_DATA, i))
                name_images.append(os.listdir(os.path.join(TRAIN_DATA, i)))
                label_images.append(i)
        except NotADirectoryError:
            print('File "data.json existed, but it was reconfigured."')

    path_name_label_zip = zip(path_images, name_images, label_images)

    # create dictionary
    train_test_image_json = {'train': {}, 'test': {}}

    # create full dict for json file
    for path_data, name_image, label in path_name_label_zip:
        for n, current_image_name in enumerate(name_image):
            if n < len(name_image) * proportion_test_images:
                train_test_image_json['test'][os.path.join(path_data, current_image_name)] = {
                 'class_name': label,
                }
            else:
                train_test_image_json['train'][os.path.join(path_data, current_image_name)] = {
                 'class_name': label,
                }
    # write json file
    with open(path_for_json, 'w') as f:
        json.dump(train_test_image_json, f, indent=4)


if __name__ == '__main__':
    make_data_json(proportion_test_images=0.2)
