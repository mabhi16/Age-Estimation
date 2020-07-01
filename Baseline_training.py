# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:39:21 2020

@author: Abhishek
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

train_imgs = np.load('train_imgs.npy')
train_labels = np.load('train_labels.npy')
val_imgs = np.load('val_imgs.npy')
val_lbls = np.load('val_lbls.npy')
train_imgs = tf.cast(train_imgs, tf.float32)
#train_imgs = (train_imgs/127.5) - 1
val_imgs = tf.cast(val_imgs, tf.float32)
#val_imgs = (val_imgs/127.5) - 1