import os
import tensorflow as tf
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model

# base_dir = 'C:\cats_and_dogs_filtered'
# 80 = 100
# 75 = 80
base_dir = r'C:\Users\josep\PycharmProjects\Fate_heroine_model'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
pred_dir = os.path.join(base_dir, 'pred')

# Directory with different heroine training pictures
train_rin_dir = os.path.join(train_dir, 'rin')
train_saber_dir = os.path.join(train_dir, 'saber')
train_sakura_dir = os.path.join(train_dir, 'sakura')

# Directory with heroine validation pictures
validation_rin_dir = os.path.join(validation_dir, 'rin')
validation_saber_dir = os.path.join(validation_dir, 'saber')
validation_sakura_dir = os.path.join(validation_dir, 'sakura')

# Set up matplotlib fig, and size it to fit 4x4 pics
import matplotlib.image as mpimg
# nrows = 4
# ncols = 4
#
# fig = plt.gcf()
# fig.set_size_inches(ncols*4, nrows*4)
# pic_index = 100
# train_cat_fnames = os.listdir( train_cats_dir )
# train_dog_fnames = os.listdir( train_dogs_dir )
#
#
# next_cat_pix = [os.path.join(train_cats_dir, fname)
#                 for fname in train_cat_fnames[ pic_index-8:pic_index]
#                ]
#
# next_dog_pix = [os.path.join(train_dogs_dir, fname)
#                 for fname in train_dog_fnames[ pic_index-8:pic_index]
#                ]
#
# for i, img_path in enumerate(next_cat_pix+next_dog_pix):
#   # Set up subplot; subplot indices start at 1
#   sp = plt.subplot(nrows, ncols, i + 1)
#   sp.axis('Off') # Don't show axes (or gridlines)
#
#   img = mpimg.imread(img_path)
#   plt.imshow(img)
#
# plt.show()

# Import the inception model
from tensorflow.keras.applications.inception_v3 import InceptionV3

pre_trained_model = InceptionV3(input_shape=(75, 75, 3),  # Shape of our images
                                include_top=False,  # Leave out the last fully connected layer
                                weights='imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > 0.959):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(pre_trained_model.output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(128, activation='relu')(x)

# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(3, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer=RMSprop(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['acc'])

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                   # rotation_range = 40,
                                   # width_shift_range = 0.2,
                                   # height_shift_range = 0.2,
                                   # shear_range = 0.2,
                                   # zoom_range = 0.2,
                                   horizontal_flip=True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1.0 / 255.)
# pred_datagen = ImageDataGenerator(rescale=1.0 / 255.)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='categorical',
                                                    target_size=(75, 75))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=20,
                                                        class_mode='categorical',
                                                        target_size=(75, 75))

# Flow test images in batches of 20 using test_datagen generator
pred_generator = test_datagen.flow_from_directory(pred_dir,
                                                  batch_size=1,
                                                  class_mode=None,  # non is used for predictions
                                                  target_size=(75, 75),
                                                  seed=1
                                                  )

# import matplotlib.pyplot as plt
# for _ in range(5):
#     img, label = train_generator.next()
#     print(label)   #  (1,256,256,3)
#     plt.imshow(img[0])
#     plt.show()

callbacks = myCallback()
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=7,
    epochs=2,
    validation_steps=20,
    verbose=1,
    callbacks=[callbacks])

# model.evaluate(validation_generator, steps=4)   # does same thing as validation data = validation_generator

# doing test brrrr
pred_generator.reset()
pred=model.predict(pred_generator, steps=44, verbose=1)

#helps see predicted answers
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# save file
filenames=pred_generator.filenames
results=pd.DataFrame({"Filename":filenames, "Predictions":predictions})
# results.to_csv("results2.csv",index=False)
# print(results.tail(10))
print(results)
print('done!')


# $ bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ~/Fate_heroine_model/resized_for_traning/
# $ bazel build tensorflow/examples/image_retraining:label_image

#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Bypass

# import matplotlib.pyplot as plt
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(len(acc))
#
# plt.plot(epochs, acc, 'bo', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
#
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training Loss')
# plt.plot(epochs, val_loss, 'b', label='Validation Loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()

# usefull command to help initate touch
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
