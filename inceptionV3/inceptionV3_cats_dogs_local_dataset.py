#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import zipfile
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

from experiment_impact_tracker.compute_tracker import ImpactTracker

tracker = ImpactTracker('inceptionTracker_02_None')
tracker.launch_impact_monitor()


# In[2]:


base_dir = '../data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')


# In[3]:


pic_index = 100
train_cat_fnames = os.listdir( train_cats_dir )
train_dog_fnames = os.listdir( train_dogs_dir )


next_cat_pix = [os.path.join(train_cats_dir, fname) 
                for fname in train_cat_fnames[ pic_index-8:pic_index] 
               ]

next_dog_pix = [os.path.join(train_dogs_dir, fname) 
                for fname in train_dog_fnames[ pic_index-8:pic_index]
               ]


# In[4]:


train_datagen = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 20, class_mode = 'binary', target_size = (224, 224))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory( validation_dir,  batch_size = 20, class_mode = 'binary', target_size = (224, 224))


# In[5]:


model = tf.keras.applications.inception_v3.InceptionV3(weights=None, include_top=False, input_shape=(244, 244, 3))


# In[14]:


for layer in model.layers:
    layer.trainable = True

x = model.layers[-1].output
x = tf.keras.layers.GlobalAveragePooling2D() (x)
predictions = tf.keras.layers.Dense(102, activation='softmax') (x)

myModel = Model(inputs=model.input, outputs=predictions)


# In[15]:


myModel.summary()


# In[16]:


myModel.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.Adam(), 
      metrics=['accuracy']
  )


# In[17]:

num_epoch = 100
inc_history = myModel.fit(
    train_generator, 
    validation_data = validation_generator, 
    steps_per_epoch = num_epoch, epochs = num_epoch)


# In[18]:


with open('myModel_pkl', 'wb') as files:
    pickle.dump(myModel, files)


# In[ ]:

score = myModel.evaluate(validation_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




