#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date:   2017-03-21
"""

from LoadDataSet import X_test, X_train, Y_train
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import *
from keras.layers.core import Dropout, Dense, Flatten, Activation
import csv

K.set_image_dim_ordering('th')  # input shape :(channels, height, width)

img_width, img_height = 28, 28

n_train = X_train.shape[0]
n_test = X_test.shape[0]

n_classes = 10
X_train = X_train.reshape(n_train, 1, img_width, img_height) / 255
X_test = X_test.reshape(n_test, 1, img_width, img_height) / 255

Y_train = to_categorical(Y_train)


n_filters = 64
filter_size1 = 3
filter_size2 = 2
pool_size1 = 3
pool_size2 = 1
n_dense = 128

model = Sequential()

model.add(Convolution2D(n_filters, filter_size1, filter_size1, batch_input_shape=(None, 1, img_width, img_height), activation='relu', border_mode='valid'))
model.add(MaxPooling2D(pool_size=(pool_size1, pool_size1)))
model.add(Convolution2D(n_filters, filter_size2, filter_size2, activation='relu', border_mode='valid'))
model.add(MaxPooling2D(pool_size=(pool_size2, pool_size2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(n_dense))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation('softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

batch_size = 128
n_epochs = 1
model.fit(X_train,
          Y_train,
          batch_size=batch_size,
          nb_epoch=n_epochs,
          verbose=2,
          validation_split=.2)

Y_pred = model.predict_classes(X_test,batch_size=32,verbose=1)

with open('CNN_submission.csv', 'wb') as MyFile:
    myWriter = csv.writer(MyFile)
    myWriter.writerow(["ImageId", "Label"])
    index = 0
    for i in Y_pred:
        tmp = []
        index = index + 1
        tmp.append(index)
        tmp.append(int(i))
        myWriter.writerow(tmp)
