#!/usr/bin/env python
# coding: utf-8

# #Transfer Learning using Inception-v3 for Image Classification

# In my [previous post](https://https://medium.com/analytics-vidhya/end-to-end-image-classification-project-using-tensorflow-46e78298fa2f), I worked on a subset of the original Dogs vs. Cats [Dataset](https://https://www.kaggle.com/c/dogs-vs-cats/data) (3000 images sampled from the original dataset of 25000 images) to build an image classifier capable of classifying images of Dogs and Cats with 82% accuracy.

# This project is in continuation with the previous one where we try to improve our performance even further.
# 
# When we have a relatively small dataset, a super effective technique is to use **Transfer Learning** where we use a pre-trained model. This model has been trained on an extremely large dataset, and we would be able to transfer weights which were learned through hundreds of hours of training on multiple high powered GPUs.
# 
# Many such models are open sourced such as VGG-19 and Inception-v3. They were trained on millions of images with extremely high computing power which can be very expensive to achieve from scratch.
# 
# We are using the **Inception-v3** model in the project.
# 
# Transfer Learning has become immensely popular because it considerably reduces training time, and requires a lot less data to train on to increase performance.

# **Get the Data** (subset)

# In[1]:


# !wget --no-check-certificate \
#     https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
#     -O /tmp/cats_and_dogs_filtered.zip


# **Import Libraries**

# In[2]:


import os
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model


base_dir = r'C:\Users\josep\PycharmProjects\Fate_heroine_model'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
pred_dir = os.path.join(base_dir, 'pred')

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'rin')

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'saber')

train_fish_dir = os.path.join(train_dir, 'sakura')

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'rin')

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'saber')
validation_fish_dir = os.path.join(validation_dir, 'sakura')



from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50

size = 75
pre_trained_model = InceptionV3(input_shape = (size, size, 3),  # Shape of our images
                                include_top = False,  # Leave out the last fully connected layer
                                weights = 'imagenet')

# pre_trained_model = ResNet50(input_shape = (75, 75, 3),  # Shape of our images
#                                 include_top = False,  # Leave out the last fully connected layer
#                                 weights = 'imagenet')




for layer in pre_trained_model.layers:
  layer.trainable = False


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.959):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True



from tensorflow.keras.optimizers import RMSprop

# output from inception v3
last_layer = pre_trained_model.get_layer('mixed7')
# x = layers.Conv2D(512, (3,3) , activation='relu')(x)
# x = layers.MaxPooling2D((2,2))(x)
x = layers.Flatten()(last_layer.output)

# output from resnet50    
# x = layers.Flatten()(pre_trained_model.output)
# last_layer = pre_trained_model.get_layer('identity_block')
# x = layers.Flatten()(last_layer.output)

x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)


# x = layers.Dense(1024, activation='relu')(x)
# x = layers.Dense(512, activation='relu')(x)
# x = layers.Dense(512, activation='relu')(x)
# x = layers.Dense(512, activation='relu')(x)
# x = layers.Dense(256, activation='relu')(x)
# x = layers.Dense(256, activation='relu')(x)

x = layers.Dropout(0.3)(x)
# Add a final sigmoid layer for classification
# x = layers.Dense  (3, activation='sigmoid')(x)    used for multi label while softmax is for multi class classification
x = layers.Dense  (3, activation='softmax')(x)

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(learning_rate=0.0001),
              # loss = 'binary_crossentropy',
              loss = 'categorical_crossentropy',
              metrics = ['acc'])


# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range =size,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )


# Flow training images in batches of 20 using train_datagen generator
batch_size = 10
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size =batch_size,
                                                    class_mode = 'categorical',
                                                    target_size = (size, size))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size  =batch_size,
                                                         class_mode  = 'categorical',
                                                         target_size = (size, size))

# Flow test images in batches of 20 using test_datagen generator
pred_generator = test_datagen.flow_from_directory(pred_dir,
                                                  batch_size=1,
                                                  class_mode=None,  # non is used for predictions
                                                  target_size=(size, size),
                                                  seed=1
                                                  )


callbacks = myCallback()
# history = model.fit(
#             train_generator,
#             validation_data = validation_generator,
#             steps_per_epoch = 2,
#             epochs = 20,#70
#             validation_steps = 2,
#             verbose = 1,
#             callbacks=[callbacks])

history = model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch =7,
            epochs = 4,
            validation_steps = 50,
            verbose = 2,
            callbacks=[callbacks])


pred_generator.reset()
pred=model.predict(pred_generator, steps=60, verbose=1)


# print(pred)
#helps see predicted answers

predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# save file
counter =0
filenames=pred_generator.filenames
results=pd.DataFrame({"Filename":filenames, "Predictions":predictions})
results["Filename"] =  results["Filename"].str.split('\\', expand = True)[0]
counterlist = results["Filename"] == results['Predictions']
print(results)
print(f"Number correct: {counterlist.sum()}")
print('done!')
# results.to_csv("results2.csv",index=False)



