# 43-1 카피

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist     #CNN을 하는 사람들은 항상 mnist를 쓴다.
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
(x_train, y_train), (x_test, y_test) = mnist.load_data()
###################### 여기부터 증폭 ######################
datagen = ImageDataGenerator(
    rescale=1./255,         # 1.은 부동소수점 형변환 1/255. 해도 된다. 어디든 .을 붙이면 된다.--------------->keras43에서 20,21행의 rescale과 같은거다.
    horizontal_flip=True,   # 수평(좌우) 뒤집기
    # vertical_flip=True,     # 수직(상하) 뒤집기    true는 한다, false는 안한다.
    width_shift_range=0.1,  # 평형이동
    # height_shift_range=0.1, 
    rotation_range=15,       # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=0.5,           
    # shear_range=0.7,        # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동(한마디로 찌부)
    fill_mode='nearest'     # 데이터가 이동 및 증폭하면 데이터가 이동한 쪽은 소실되고 그전에 있던 곳은 비워지게 된다. 비워진 곳은 새롭게 채워야 하는데 그때 근처에 있는 수치로 채운다.
)

augment_size=40000

randidx = np.random.choice(x_train.shape[0], size=augment_size, replace=False)     

x_augmented = x_train[randidx].copy()        
y_augmented = y_train[randidx].copy()


x_augmented = x_augmented.reshape(                              
        x_augmented.shape[0],
        x_augmented.shape[1],
        x_augmented.shape[2], 1)

x_augmented = datagen.flow(
                x_augmented, y_augmented,
                batch_size=augment_size,
                shuffle=False,
).next()[0]

print(x_augmented.shape)       #(40000, 28, 28, 1)

print(x_train.shape)        #(60000, 28, 28)

# exit()
x_train = x_train/255.
x_test = x_test/255.


x_train = np.concatenate((x_train, x_augmented))          
y_train = np.concatenate((y_train, y_augmented))


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
# y_train = y_train.reshape(60000, 1)
y_train = y_train.reshape(-1, 1) # y의 범위를 모를땐 전체를 reshape한다.
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)



#2. 모델구성
model = Sequential() 
input1 = Input(shape=(28, 28, 1))

conv1_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input1)
conv1_2 = Conv2D(64, (3, 3), activation='relu')(conv1_1)
pool1   = MaxPooling2D()(conv1_2)
drop1   = Dropout(0.2)(pool1)

conv2_1 = Conv2D(128, (3, 3), padding='same', activation='relu')(drop1)
conv2_2 = Conv2D(128, (3, 3), activation='relu')(conv2_1)
pool2   = MaxPooling2D()(conv2_2)
drop2   = Dropout(0.25)(pool2)

conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu')(drop2)
conv3_2 = Conv2D(256, (3, 3), activation='relu')(conv3_1)
pool3   = MaxPooling2D()(conv3_2)
drop3   = Dropout(0.3)(pool3)

gap    = GlobalAveragePooling2D()(drop3)
dense1  = Dense(512, activation='relu')(gap)
drop4   = Dropout(0.3)(dense1)
dense2  = Dense(256, activation='relu')(drop4)
drop5   = Dropout(0.2)(dense2)

output1 = Dense(10, activation='softmax')(drop5)
model   = Model(inputs=input1, outputs=output1)




# exit()
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=15,
        restore_best_weights=True,
)

start_time=time.time()
hist = model.fit(x_train, y_train, epochs=150, batch_size=750,
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


# 0.995 맞추기



'''
keras43 best
loss : 0.02006814070045948
acc : 0.9945999979972839
accuracy_score :  0.9946
걸린시간 :  133.37 초

1차
loss : 0.019431430846452713
acc : 0.995199978351593
accuracy_score :  0.9952
걸린시간 :  383.21 초










'''






