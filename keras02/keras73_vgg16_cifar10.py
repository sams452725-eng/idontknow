import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
import random
import numpy as np
import pandas as pd
import os
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Input
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
###################### 여기부터 증폭 ######################
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    # height_shift_range=0.1, 
    rotation_range=15,
    # zoom_range=0.5,           
    # shear_range=0.7,
    fill_mode='nearest'
)

augment_size=40000

randidx = np.random.choice(x_train.shape[0], size=augment_size, replace=False)     

x_augmented = x_train[randidx].copy()        
y_augmented = y_train[randidx].copy()


x_augmented = x_augmented.reshape(                              
        x_augmented.shape[0],
        x_augmented.shape[1],
        x_augmented.shape[2], 
        3)

x_augmented = datagen.flow(
                x_augmented, y_augmented,
                batch_size=augment_size,
                shuffle=False,
).next()[0]

x_train = x_train/255.
x_test = x_test/255.

x_train = np.concatenate((x_train, x_augmented))          
y_train = np.concatenate((y_train, y_augmented))

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
# y_train = y_train.reshape(60000, 1)
y_train = y_train.reshape(-1, 1) 
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)


random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)


#2. 모델
from keras.applications import VGG16
from tensorflow.keras.datasets import cifar10

vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32,32,3),
)
vgg16_trainable=False
# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.005
# learning_rate = 0.009

from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping


model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate),
              metrics=['acc'])
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=14,
        restore_best_weights=True,
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5,        
)  
start_time=time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=128,
          verbose=1,
          validation_split=.2,
          callbacks=[es,]
          )
end_time = time.time()


#4. 평가,예측
print('================= model.evaluate =================')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)
y_test = np.argmax(y_test, axis=1).reshape(-1, 1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time - start_time, 2), '초')


# 실습 #
# 비교할거
# 1. 이전에 실해한 최상의 결과
# 2. 가중치를 동결하고 훈련시켰을 때, vgg16_trainable=False
# 3. 가중치를 동결하지 않고 훈련시켰을 때, vgg16_trainable=True
# 4. 시간까지 비교.


'''
keras53 best
loss : 0.6173158884048462
acc : 0.850600004196167
accuracy_score :  0.8506
걸린시간 :  1068.12 초

vgg16_trainable=False
loss : 1.6483126878738403
acc : 0.3873000144958496
accuracy_score :  0.3873
걸린시간 :  398.35 초

vgg16_trainable=True






'''