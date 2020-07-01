# -*- coding: utf-8 -*-
"""
@author:Base code: https://github.com/LokLu/Tensorflow-data-loader
@author:Customized code: Abhishek Murali
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import data_loader_git

data_list = 'G:/Deep Learning/Age_detection/img_adr.txt'
val_list = 'G:/Deep Learning/Age_detection/val_data.txt'
plt.ioff()

# Parse the images and masks, and return the data in batches, augmented optionally
data = data_loader_git.data_batch(data_list, augment=[], normalize=True, batch_size=20, epoch=None)
val_data = data_loader_git.data_batch(val_list, augment=[], normalize=True, batch_size=10, epoch=None)
# Get the image and mask op from the returned dataset
image_tensor, mask_tensor = data
val_img, val_lbl = val_data

# with tf.Session() as sess:
    # Evaluate the tensors
    # for i in range(1):
        # image, mask = sess.run([image_tensor, mask_tensor])
        # v_img, v_lbl = sess.run([val_img, val_lbl])
        # print(mask)
        # print(v_lbl)
        # Confirming everything is working by visualizing
        # plt.figure('train image')
        # plt.imshow(image[0, :, :, :])
        # plt.figure('augmented mask')
        # plt.imshow(mask[0, :, :,:])
        # plt.show()
        # plt.figure('Val image')
        # plt.imshow(v_img[0, :, :, :])
        # plt.figure('augmented mask')
        # plt.imshow(mask[0, :, :,:])
        # plt.show()
        # Do whatever you want now, like creating a feed dict and train your models

IMG_SIZE = 200
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
# feature_batch = base_model()
# print(feature_batch.shape)
base_model.trainable = False
# base_model.summary()
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# feature_batch_average = global_average_layer(feature_batch)
# print(feature_batch_average.shape)
prediction_layer = tf.keras.layers.Dense(9)
# prediction_batch = prediction_layer(feature_batch_average)
# print(prediction_batch.shape)
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

epochs_number = 10
# Fitting the model with train batches
history = model.fit(
    image_tensor, mask_tensor, batch_size=32,
    epochs=epochs_number, verbose=1,
    validation_data=(val_img, val_lbl), steps_per_epoch=77, validation_steps=200
    )
