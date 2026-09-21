# 36-2 카피

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist     #CNN을 하는 사람들은 항상 mnist를 쓴다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) 
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# print(np.max(x_train), np.min(x_train)) # 255 0    ------> 일반적으로 이미지는 최대값이 255이기 때문에 굳이 print를 안해봐도 쉽게 스케일링을 할수있다. 일반적인 이미지의 최대값이 255인 이유는 픽셀의 명암수치가 0~255사이이기 때문이다.
# print(np.max(x_test), np.min(x_test))   # 255 0

###### 스케일링 1 ######
x_train = x_train/255. # 뒤에 '.' 을 붙이는 이유는 산출된 값을 파이썬의 float 형식으로 바꿔주기 위함이다.(파이썬 기초)
x_test = x_test/255. 
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test))   # 1.0 0.0

###### 스케일링 2 ######
# x_train = (x_train-127.5)/127.5        
# x_test = (x_test-127.5)/127.5
# print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
# print(np.max(x_test), np.min(x_test))   # 1.0 -1.0

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
# print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

# print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64)) -----> 0이 5923개, 1이 6742개 등등

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
# y_train = y_train.reshape(60000, 1)
y_train = y_train.reshape(-1, 1) # y의 범위를 모를땐 전체를 reshape한다.
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)



#2. 모델구성
model = Sequential() #파라미터의 개수는 w와 b의 개수다. dropout은 파라미터에 영향이 없다.
model.add(Conv2D(64, (5,5), input_shape=(28,28,1)))  #(26,26,64)
model.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu'))  #(24,24,32)
model.add(Dropout(.2))
model.add(Conv2D(32, (3,3), activation='relu'))  #(23,23,32)
model.add(Conv2D(16, (2,2), activation='relu'))  #(22,22,16)
model.add(Dropout(.2))
model.add(Conv2D(16, (2,2), activation='relu'))  #(21,21,16)
model.add(Dropout(.2))
model.add(Conv2D(16, (1,1), activation='relu'))  #(20,20,16)
# model.add(Flatten())  #( , 64000) -----> reshape와 동일한 기능으로 행 부분을 제외하고 열 부분을 2차원으로 만들어서 Dense와 바로 엮이게 만들어준다.
model.add(GlobalAveragePooling2D()) 
# 결국 마지막 반환값은 100개인데, Flatten을 쓰면 (20*20*16)개가 내려와서 Dense를 10만 줘도 640000개로 10개를 찾으니 연산도 오래되고 데이터가 너무 많다.
# 근데 GlobalAveragePooling을 하면 앞에 20*20을 평균내서 1로 만들고 뒤에오는 channel값만 가지고 값을 조정할수있다.
# 그러면 밑에 Dense와 만나서 연산되는 파라미터 자체가 확연히 줄어서 결과값이 더 좋게 나오곤한다. 그래서 Flatten보다 GlobalAveragePooling을 쓴다.
model.add(Dense(units=32, activation='relu'))  # units는 filters와 같이 output node의 개수를 일컫는 말이다.
model.add(Dropout(.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 마지막 Dense에서는 결국 (, 10)의 shape이 되어야 하는데, conv상태에서는 ( , , )형태이기 때문에 summary까지는 나오는데 이후 단계에서 error가 뜬다.
# model.summary()
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 26, 26, 64)        640       
                                                                 
#  conv2d_1 (Conv2D)           (None, 24, 24, 32)        18464     
                                                                 
#  dropout (Dropout)           (None, 24, 24, 32)        0         
                                                                 
#  conv2d_2 (Conv2D)           (None, 23, 23, 32)        4128      
                                                                 
#  conv2d_3 (Conv2D)           (None, 22, 22, 16)        2064      
                                                                 
#  dropout_1 (Dropout)         (None, 22, 22, 16)        0         
                                                                 
#  conv2d_4 (Conv2D)           (None, 21, 21, 16)        1040      
                                                                 
#  dropout_2 (Dropout)         (None, 21, 21, 16)        0         
                                                                 
#  conv2d_5 (Conv2D)           (None, 20, 20, 16)        1040      
                                                                 
#  flatten (Flatten)           (None, 6400)              0         
                                                                 
#  dense (Dense)               (None, 32)                204832    
                                                                 
#  dropout_3 (Dropout)         (None, 32)                0         
                                                                 
#  dense_1 (Dense)             (None, 16)                528       
                                                                 
#  dense_2 (Dense)             (None, 10)                170    


# exit()
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

start_time=time.time()
model.fit(x_train, y_train, epochs=50, batch_size=128,
          verbose=1,
          validation_split=.2,
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
py311
loss : 0.04604989290237427
acc : 0.9897000193595886
accuracy_score :  0.9897
걸린시간 :  545.91 초

tf29x-gpu
loss : 0.03711336851119995
acc : 0.9904999732971191
accuracy_score :  0.9905
걸린시간 :  135.76 초


py311
loss : 0.03212791308760643
acc : 0.9918000102043152
accuracy_score :  0.9918
걸린시간 :  711.91 초

'''






