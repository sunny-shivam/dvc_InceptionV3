#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# 
# ## Install TensorFlow 2
# 

# In[ ]:


import tensorflow as tf
tf.__version__


# In[ ]:


from experiment_impact_tracker.compute_tracker import ImpactTracker


# In[ ]:




# 
# ## Import other libraries
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

np.set_printoptions(precision=7)
# get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow_datasets as tfds

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from collections import Counter
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image

from PIL import Image


tracker = ImpactTracker('tracker_03')
tracker.launch_impact_monitor()
# ## Create Directory for Dataset

# In[ ]:


import os
import errno


try:
    data_dir = 'dataset'
    os.mkdir(data_dir)
except OSError as e:
    if e.errno == errno.EEXIST:
        print('Directory  created.')
    else:
        raise


# ---
# # TFDS Datasets
# 

# ## Choose Dataset Cifar10

# In[ ]:


dataset_name = "cifar10"


# 
# ## Download Dataset
# 

# In[ ]:


(train_set, test_set), dataset_info =  tfds.load( 
              name=dataset_name, 
              split=["train", "test"], 
              with_info=True, 
              data_dir=data_dir
          )


# ## Dataset Information

# In[ ]:


print(dataset_info)


# ### Detailed Information

# In[ ]:


class_names =  dataset_info.features["label"].names

print('image shape    :', dataset_info.features['image'].shape)
print('image dtype    :', dataset_info.features['image'].dtype)
print()
print('num class      : ',dataset_info.features["label"].num_classes)
print('class label    :', dataset_info.features["label"].names)
print()
print('num train data :', dataset_info.splits["train"].num_examples)
print('num test data  :', dataset_info.splits["test"].num_examples)


# ## Show Images

# # Preprocess Image
# 
# Convert and Resize Dataset to Numpy

# In[ ]:


input_shape = (80, 80, 3)


# ### Convert Data Train

# In[ ]:


X_train = []
y_train = []

for example in tfds.as_numpy(train_set):
    new_img = example['image']
    new_img = cv.resize(new_img, input_shape[:2],interpolation = cv.INTER_AREA) 
    X_train.append(new_img)
    y_train.append(example['label'])

del train_set


# In[ ]:


X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

print('X_train.shape =',X_train.shape)
print('y_train.shape =',y_train.shape)


# ### Convert Data Test

# In[ ]:


X_test = []
y_test = []

for example in tfds.as_numpy(test_set):
    new_img = example['image']
    new_img = cv.resize(new_img, input_shape[:2],interpolation = cv.INTER_AREA) 
    X_test.append(new_img)
    y_test.append(example['label'])

del test_set


# In[ ]:


X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

print('X_test.shape =',X_test.shape)
print('y_test.shape =',y_test.shape)


# ## Split Data Train into Train and Val

# In[ ]:


X_val   = X_train[-300:]
y_val   = y_train[-300:]

X_train = X_train[:-300]
y_train = y_train[:-300]


# In[ ]:


print('X_train.shape =',X_train.shape)
print('y_train.shape =',y_train.shape)

print('\nX_val.shape  =',X_val.shape)
print('y_val.shape  =',y_val.shape)

print('\nX_test.shape  =',X_test.shape)
print('y_test.shape  =',y_test.shape)


# # One hot y labels

# In[ ]:


y_train_hot = to_categorical(y_train, 102)
y_val_hot   = to_categorical(y_val, 102)
y_test_hot  = to_categorical(y_test, 102)

print('y_train_hot.shape =',y_train_hot.shape)
print('y_val_hot.shape   =',y_val_hot.shape)
print('y_test_hot.shape  =',y_test_hot.shape)


# ---
# # Classification Model

# ## Create Model
# 
#  Using inception-resnet-v2

# In[ ]:


# model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(80, 80, 3))
model = tf.keras.applications.inception_v3.InceptionV3(weights=None, include_top=False, input_shape=(80, 80, 3))

for layer in model.layers:
    layer.trainable=True

# In[ ]:


x = model.layers[-1].output
x = GlobalAveragePooling2D() (x)
predictions = Dense(102, activation='softmax') (x)

myModel = Model(inputs=model.input, outputs=predictions)


# ## Visualize Model

# In[ ]:


myModel.summary()


# ## Compile Model
# 
# 

# In[ ]:


myModel.compile(
      loss='categorical_crossentropy',
      optimizer=tf.keras.optimizers.Adam(), 
      metrics=['accuracy']
  )


# ---
# # Data Augmentation

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    channel_shift_range=0.1,
    horizontal_flip=True
)


# #  Callbacks
# * Checkpoint
# * Learning Rate Annealing

# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

def lr_schedule(epoch):
    lr = 1e-3
    if (epoch > 30):
        lr *= 0.01
    elif (epoch > 20):
        lr *= 0.1
    return lr

lr_callback = LearningRateScheduler(lr_schedule)

myCheckpoint = ModelCheckpoint(filepath='./dataset/my_model.h5', 
                               monitor='val_accuracy',
                               save_best_only=True,
                              )


# # Train the Model

# In[ ]:


history_all = []


# In[ ]:


batch_size = 64
epochs = 100


# In[ ]:


augmented_train = datagen.flow(
    X_train, y_train_hot, batch_size
)

history = myModel.fit(
    augmented_train,
    validation_data=(X_val, y_val_hot),
    epochs=epochs, 
    steps_per_epoch=len(X_train)/64,
    callbacks=[lr_callback, myCheckpoint],
    verbose=2)

history_all.append(history)


# ## Plot Current History Training

# # Evaluate Model
# 

# In[ ]:


myModel.load_weights('./dataset/my_model.h5')
scores = myModel.evaluate(X_test, y_test_hot)


# In[ ]:


print('Test loss    :', scores[0])
print('Test accuracy: %.2f%%' % (scores[1]*100))


# # Re-Evaluate Model
# 

# In[ ]:


train_scores = myModel.evaluate(X_train, y_train_hot)
test_scores  = myModel.evaluate(X_test, y_test_hot)
val_scores   = myModel.evaluate(X_val, y_val_hot)


# In[ ]:


print('Train Loss: %.5f with Accuracy: %.1f%%' % (train_scores[0], (train_scores[1]*100)))
print('Test  Loss: %.5f with Accuracy: %.1f%%' % (test_scores[0], (test_scores[1]*100)))
print('Val   Loss: %.5f with Accuracy: %.1f%%' % (val_scores[0], (val_scores[1]*100)))

