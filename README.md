Aim of this project is compare the performance of various exsisting neural networks in estimating the age based on the facial features. The dataset for this project was downloaded from opensource Kaggle dataset: https://www.kaggle.com/frabbisw/facial-age. 

Neural Networks under consideration:
MobileNetV2,
VGG16,
Resnet-50,
Inception,

The data provided by kaggle had ages ranging from 1 to 110 as individual classes, for the convinence i have grouped them into 9 classes: 1 to 10, 11 to 20, 21 to 30, 31 to 40, 41 to 50, 51 to 60, 61 to 70, 71 to 80, 81 to 100.

As of now the i have visualized the data distributon among different classes on a pie chart, determined the class with min data and have taken same number of images from each class to establish a baseline performance of the network. Currently i am working on the MobileNetV2 baseline performance test. 

Data Loader base code has been taken from : https://github.com/LokLu/Tensorflow-data-loader, since this data loader was written in Python 2 and it was designed for Image segmentation, i have customized few parts of the code and added a function for integer encoding of labels.


