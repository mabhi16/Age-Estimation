Aim of this project is compare the performance of various exsisting neural networks in estimating the age based on the facial features. The dataset for this project was downloaded from opensource Kaggle dataset: https://www.kaggle.com/frabbisw/facial-age. 

Neural Networks under consideration:
MobileNetV2,
VGG16,
Resnet-50,
Inception,

The data provided by kaggle had ages ranging from 1 to 110 as individual classes, for the convinence i have grouped them into 9 classes: 1 to 10, 11 to 20, 21 to 30, 31 to 40, 41 to 50, 51 to 60, 61 to 70, 71 to 80, 81 to 100.

As of now the i have visualized the data distributon among different classes on a pie chart, determined the class with min data and have taken same number of images from each class to establish a baseline performance of the network. Currently i am working on the MobileNetV2 baseline performance test. 

Data Loader base code has been taken from : https://github.com/LokLu/Tensorflow-data-loader, since this data loader was written in Python 2 and it was designed for Image segmentation, i have customized few parts of the code and added a function for integer encoding of labels.

As it can be seen from the accuracy measure graph that the model is overfitting, i am currently working on hyper-parameter tuning to solve the overfitting problem.

Update On 23/07/2020 : 
After the hyper-parameter tuning i found out the major problem would be the problem in the classification is the similar features being present in the data of different classes. For example the face features of person at 50 and 51 are very similar, but in our problem the person in 50 and person in 51 both fall into two different classes. so i had reduced the problem into a binary classification to examine if the validation accuracy increases, the validation accuracy increased rapidly from 25% to 88%. So i will try to break this classes further into small chuncks like 4 class problem and further or take a regression approach to predict a definite age instead of a window like 1 to 10.


