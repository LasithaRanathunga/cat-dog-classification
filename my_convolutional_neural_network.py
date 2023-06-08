# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:58:28 2023

@author: Lenovo
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#innitializing the cnn
classifire = Sequential()

# spet-1 - cinverlution
classifire.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation= 'relu'))

# step 2 - pooling
classifire.add(MaxPooling2D(pool_size= (2, 2)))

# step 3 - flattening
classifire.add(Flatten())

# step 4 - full connection
classifire.add(Dense(units = 128 , activation= 'relu'))
classifire.add(Dense(units = 1 , activation= 'sigmoid'))

# compiling the cnn
classifire.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# fitting the CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifire.fit(training_set,
                        steps_per_epoch=625,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=156)