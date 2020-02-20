import tensorflow as tf 

import matplotlib.pyplot as plt
import numpy as np


from keras.datasets import mnist

import os
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)

KTF.set_session(session)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense , Dropout,Conv2D, MaxPooling2D, Flatten

# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()


# rescale to have values within 0 - 1 range [0,255] --> [0,1]
X_train = trainX.astype('float32')/255
X_test = testX.astype('float32')/255 

# build the model object
model = Sequential()


#define Model
# CONV_1: add CONV layer with RELU activation and depth = 32 kernels
model.add(Conv2D(32, kernel_size=(3, 3), padding='same',activation='relu',input_shape=(28,28,1)))
# POOL_1: downsample the image to choose the best features 
model.add(MaxPooling2D(pool_size=(2, 2)))
# CONV_2: here we increase the depth to 64
model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
# POOL_2: more downsampling
model.add(MaxPooling2D(pool_size=(2, 2)))
# flatten since too many dimensions, we only want a classification output
model.add(Flatten())
# FC_1: fully connected to get all relevant data
model.add(Dense(64, activation='relu'))
# FC_2: output a softmax to squash the matrix into output probabilities for the 10 classes
model.add(Dense(10, activation='softmax'))
model.summary()

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])

# load the weights that yielded the best validation accuracy
model.load_weights('model.weights.best.hdf5')    

#take no. 117 image from test set
img = X_test[117]
#reshape image into single channel, 28x28, single image
test_img = img.reshape((1,28,28,1))

img_class = model.predict_classes(test_img)
prediction = img_class[0]
classname = img_class[0]
print("Class: ",classname)

img = img.reshape((28,28))
plt.imshow(img)
plt.title(classname)
plt.show()