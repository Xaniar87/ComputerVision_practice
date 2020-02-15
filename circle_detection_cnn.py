import numpy as np
from numpy import newaxis
from skimage import io
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt

def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
        mary eslanadari
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def data(noise,n):
    params_array, images = noisy_circle(200, 50, noise)
    images=[images]
#     new=(img-np.min(img))/(np.max(img)-np.min(img))*255
    params_array=[params_array]
    for _ in range(1,n):
        params, img = noisy_circle(200, 50, 2)
        images=np.concatenate((images,[img]),axis=0)
        params_array=np.concatenate((params_array,[params]),axis=0)
    return  params_array, images
y_train, x_train=data(0,10000)
y_test, x_test=data(0,1000)
x_train=x_train[:,:,:,newaxis]
y_train=y_train[:,0:2]
x_test=x_test[:,:,:,newaxis]
y_test=y_test[:,0:2]


from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
size=200
batch_size=128
epochs=5
input_shape = (size,size,1)
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),activation='linear',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation='linear'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation='linear'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation='linear'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation='linear'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))

model.add(Conv2D(256, (3, 3), activation='linear'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))


model.add(Flatten())
# model.add(Dropout(0.5))
model.add(Dense(128, activation='linear'))
# model.add(Dropout(0.5))
model.add(Dense(2, activation='linear'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
