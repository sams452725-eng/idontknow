# 실습
# acc 기준 .93 이상

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist     
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

# print(np.max(x_train), np.min(x_train)) #255 0

###### 스케일링 1 ######
x_train = x_train/255. 
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
y_train = y_train.reshape(-1, 1) 
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)



#2. 모델구성
model = Sequential() 
model.add(Conv2D(64, (5,5), input_shape=(28,28,1))) 
model.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu'))  
model.add(Dropout(.3))
model.add(Conv2D(32, (3,3), activation='relu'))  
model.add(Conv2D(16, (2,2), activation='relu'))  
model.add(Dropout(.3))
model.add(Conv2D(16, (2,2), activation='relu'))  
model.add(Dropout(.3))
model.add(Conv2D(16, (1,1), activation='relu'))  
model.add(Flatten())  

model.add(Dense(units=32, activation='relu'))  
model.add(Dropout(.3))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(10, activation='softmax')) 
# model.summary()

# exit()
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        restore_best_weights=True,
)
start_time=time.time()
model.fit(x_train, y_train, epochs=50, batch_size=50,
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



''''
1차
loss : 0.3461201786994934
acc : 0.9070000052452087
accuracy_score :  0.907
걸린시간 :  119.28 초

2차
loss : 0.2845168709754944
acc : 0.9013000130653381
accuracy_score :  0.9013
걸린시간 :  207.9 초

3차
loss : 0.2682419717311859
acc : 0.906000018119812
accuracy_score :  0.906
걸린시간 :  199.42 초

4차
loss : 0.3151441216468811
acc : 0.890500009059906
accuracy_score :  0.8905
걸린시간 :  98.32 초

5차
loss : 0.31689682602882385
acc : 0.8942000269889832
accuracy_score :  0.8942
걸린시간 :  108.28 초

6차
loss : 0.3358216881752014
acc : 0.8776999711990356
accuracy_score :  0.8777
걸린시간 :  91.26 초

7차
loss : 0.5181727409362793
acc : 0.8008999824523926
accuracy_score :  0.8009
걸린시간 :  112.68 초

8차
loss : 0.5362592935562134
acc : 0.7946000099182129
accuracy_score :  0.7946
걸린시간 :  107.96 초

9차
loss : 0.7169229984283447
acc : 0.7402999997138977
accuracy_score :  0.7403
걸린시간 :  113.45 초

10차
loss : 0.6402547359466553
acc : 0.753600001335144
accuracy_score :  0.7536
걸린시간 :  196.85 초

11차
loss : 0.2804967761039734
acc : 0.9041000008583069
accuracy_score :  0.9041
걸린시간 :  465.75 초

12차
loss : 0.3563579320907593
acc : 0.8672000169754028
accuracy_score :  0.8672
걸린시간 :  230.69 초
'''








