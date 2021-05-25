# cats_dogs_detection

Scripts for training and testing convolutional neural network for localization of the object.

This script builds models using these libraries "https://github.com/qubvel/classification_models" and 
"https://github.com/qubvel/efficientnet". Using these libraries made it possible to quickly test various models and 
choose the best option for solving the localization problem.

You can view or set parameters in config.py

To train the model, a dataset consisting of 1037 images of cats and 2348 images of dogs was used, 3385 in total. 
This dataset consists of the "Oxford IIIT Pet Dataset".
> https://www.robots.ox.ac.uk/~vgg/data/pets/

## 1. Dataset preparation
My dataset has this configuration:
```
cats_dogs_dataset/
             train/
                Abyssinian_1.jpg
                Abyssinian_1.txt
                ...
             valid/
                Abyssinian_116.jpg
                Abyssinian_116.txt
                ...
``` 

## 1. Training
Run training script with default parameters:
```shell script
python train.py
```
## 2. Plotting graphs
If you want to build graphs from saved logs, you can use tens or board by passing the path to the logs folder.
```shell script
tensorboard --logdir models_data/tensorboard_logs/resnet18_imagenet_2021-05-23_15-19-49
```
## 3. Testing
If you want to use webcam for the test, you should pass two arguments --weights and --webcam. The default value is True 
for weights.
```shell script
python test.py --weights models_data/save_models/Linknet_imagenet_2021-05-18_18-42-54_True/Linknetefficientnetb0.h5 --webcam
```
If you want to calculate the average value of losses and indicators on val_data, you should use --metrics.
```shell script
python test.py --metrics
```
If you want to calculate the average inference time and average fps and indicators on val_data, you should use --time.
```shell script
python test.py --time
```
If you want to use gpu for test, you should use --gpu.
```shell script
python test.py --gpu
```
## Results
The table displays the resulting metrics and other information on a validation dataset consisting of 799 images. All 
models used weights "imagenet" and the Adam optimizer with a learning rate 0.0001. These results are averaged.

|    Name Model      | Loss     | Accuracy |   IoU   | Mean inference time | Mean FPS | image shape |
|:------------------:|:--------:|:--------:|:-------:|:-------------------:|:--------:|:-----------:|
|resnet18            | 0.4279   | 0.9821   | 0.9167  | 0.0212              | 47.11    |(224, 224, 3)|
|resnet18            | 0.4333   | 0.9844   |0.9406   |0.0213               |46.86     |(256, 256, 3)|
|resnet18            | 0.4240   |0.9953    |0.9354   |0.0218               |45.84     |(288, 288, 3)|
|resnet18            | 0.4329   |0.9814    |0.9369   |0.0226               |44.17     |(384, 384, 3)|
|resnet34            | 0.4353   |0.9865    |0.9342   |0.0247               |40.44     |(256, 256, 3)|
|resnet34            | 0.4490   |0.9678    |0.8644   |0.0280               |35.65     |(512, 512, 3)|
|mobilenetv2         | 0.4266   |0.9916    |0.9109   |0.0229               |43.68     |(224, 224, 3)|
|EfficientNetB0      | 0.4311   |0.9910    |0.9349   |0.0275               |36.41     |(224, 224, 3)|
|densenet121         | 0.4168   |0.9957    |0.9323   |0.0301               |33.22     |(224, 224, 3)|
