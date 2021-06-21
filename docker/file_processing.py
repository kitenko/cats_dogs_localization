import os
import time
import argparse
from typing import Tuple, List, Dict

import cv2
import numpy as np
import onnxruntime as rt
from tqdm import tqdm


def input_args(input_argument: str, model_path: str) -> None:
    """
    This function, depending on the input argument, reproduces a certain operation scenario. If '0' is passed as an
    argument, then the input data from the webcam will be used, if a video file is passed, then the video file will be
    processed, if the path to the folder is passed, in this case the images in the folder will be processed.

    :param input_argument: This is an input parameter, depending on which a certain script will be executed.
    :param model_path: This is path for load onnx model.
    """

    if os.path.isfile(input_argument):
        visualization(path_file=input_argument, model_path=model_path, mode='video processing',
                      path_folder=input_argument)

    elif os.path.isdir(input_argument):
        visualization(path_folder=input_argument, model_path=model_path, mode='images processing',
                      path_file=input_argument)

    else:
        try:
            input_argument = int(input_argument)
            if type(input_argument) is int:
                visualization(path_folder=str(input_argument), model_path=model_path, mode='-',
                              path_file=str(input_argument))
        except ValueError:
            print('If you want to use a webcam for prediction, you need to use id for webcam.')


def preparing_frame(image: np.ndarray, model: rt.InferenceSession, cam_width: float,
                    cam_height: float) -> Tuple[List[int], int]:
    """
    This function prepares the image and makes a prediction.

    :param cam_height: height of the input image from the webcam.
    :param cam_width: width of the input image from the webcam
    :param image: this is input image or frame.
    :param model: this is a loaded model.
    :return: bounding_box and class for image.
    """
    image = cv2.resize(image, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
    image = np.float32(np.expand_dims(image, axis=0) / 255.0)
    input_name = model.get_inputs()[0].name
    predict = model.run(None, {input_name: image})[0]
    predict = predict[0]
    if predict[0] <= 0.5:
        label = 0
    else:
        label = 1

    bounding_box = [int(predict[1]*cam_width), int(predict[2]*cam_height),  int(predict[3]*cam_width),
                    int(predict[4]*cam_height)]

    return bounding_box, label


def visualization(model_path: str, mode: str, path_file: str, path_folder: str) -> None:
    """
    This function, depending on the input parameter 'mode', processes the input data differently.
    """
    cat_dog = {0: 'cat', 1: 'dog'}
    box_color = (255, 0, 0)
    text_color = (255, 255, 255)

    model = rt.InferenceSession(model_path)

    if mode == 'images processing':

        path_save_image = os.path.join('file_share', os.path.basename(path_folder) + '_processed')
        os.makedirs(path_save_image, exist_ok=True)
        images = os.listdir(path_folder)

        for i in tqdm(range(len(images))):
            img = cv2.imread(os.path.join(path_folder, images[i]))
            if img is None:
                print('There are problems with the image file. \n Path: ' + os.path.join(path_folder, images[i]))
                continue
            width = img.shape[1]
            height = img.shape[0]

            bounding_box, label = preparing_frame(image=img, model=model, cam_width=width, cam_height=height)

            x_min, y_min, x_max, y_max = bounding_box
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
            cv2.imwrite(os.path.join(path_save_image, images[i]), img)

        return

    elif mode == 'video processing':
        cap = cv2.VideoCapture(path_file)
        path_save_video = os.path.join('file_share', os.path.basename(os.path.splitext(path_file)[0]) + '_processed')

    else:
        cap = cv2.VideoCapture(int(path_file))
        path_save_video = os.path.join('file_share', path_file)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    count_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(path_save_video, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(path_save_video, 'output_video.avi'), fourcc, fps, (int(width), int(height)))

    prev_frame_time = 0
    print('To exit, press "Ctrl-C"')

    if mode == 'video processing':
        for _ in tqdm(range(count_frame)):
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            video_processing(cap, model, width, height, box_color, cat_dog, text_color, out, fps)
            prev_frame_time = new_frame_time
    else:
        while cap.isOpened():
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            video_processing(cap, model, width, height, box_color, cat_dog, text_color, out, fps)
            prev_frame_time = new_frame_time
    cap.release()
    out.release()


def video_processing(cap: cv2.VideoCapture, model: rt.InferenceSession, width: int, height: int,
                     box_color: Tuple[int, int, int], cat_dog: Dict, text_color: Tuple[int, int, int],
                     out: cv2.VideoWriter, fps: float) -> None:
    """
    This function reads the frame, makes a prediction, and draws a rectangle, recording each frame.
    """
    ret, frame = cap.read()
    bounding_box, label = preparing_frame(image=frame, model=model, cam_width=width, cam_height=height)
    x_min, y_min, x_max, y_max = bounding_box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=box_color, thickness=2)
    ((text_width, text_height), _) = cv2.getTextSize(cat_dog[label],
                                                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(frame, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), box_color, -1)
    cv2.putText(
        frame,
        text=cat_dog[label],
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=text_color,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(frame, str(int(fps)) + ':fps', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    out.write(frame)


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('script for model testing.')
    parser.add_argument('--model_onnx', type=str, default='onnx_model.onnx', help='Path for loading model weights.')
    parser.add_argument('--input_shape', type=Tuple, default=(256, 256, 3), help='inpute shape model')
    parser.add_argument('--input', type=str, default=None, help='input')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    INPUT_SHAPE = args.input_shape
    input_args(input_argument=args.input, model_path=args.model_onnx)
