
## Identification Of Natural Scenes 
This project leverages the Intel Scene Classification dataset sourced from Kaggle, comprising training, testing, and prediction datasets. The implementation revolves around a deep learning model known as Convolutional Neural Network (CNN).

## About 
The dataset consists of approximately 25,000 images, each sized at 150x150 pixels, and categorized into six distinct classes: {'buildings' -> 0, 'forest' -> 1, 'glacier' -> 2, 'mountain' -> 3, 'sea' -> 4, 'street' -> 5}. The data is partitioned into three sets: Training (14,000 images), Testing (3,000 images), and Prediction (7,000 images). Originally, Intel released this data on https://datahack.analyticsvidhya.com to host an Image Classification Challenge.

## About Convolutional Neural Network (CNN):
A Convolutional Neural Network (CNN) is a deep learning architecture specifically designed for processing structured grid data, such as images. It excels in capturing spatial hierarchies through the application of convolutional filters, enabling the extraction of meaningful features from input images. CNNs are widely used in computer vision tasks, including image classification, object detection, and segmentation, due to their ability to automatically learn hierarchical representations of visual data.

## Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

import os
from os import walk

## Testing and Training
tf.keras.layers.Conv2D(filters=24, kernel_size=(5,5), activation='relu', padding='same')

hist = mod.fit(train_ds, epochs=50, batch_size=batch, verbose=1, shuffle=shuff, validation_data=val_ds, callbacks=[earlystop, lr])
   
## Result
Using different size and number of kernels and different number of epoch gave different accuracy.
using epoch= 50, filters=24, kernel_size=(5,5) gave accuracy score of 17%

using epoch =50, filters=16, kernel_size=(3,3) gave accuracy score of 75%

using epoch =3, filters=16, kernel_size=(3,3) gave accuracy score of 57%

## Accuracy ConfusionMatrixDisplay
![image](https://github.com/amna55/Identification-of-Natural-Scenes/assets/106149828/355d62dd-f6a0-4820-9cd6-7d557604026b)


## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)

