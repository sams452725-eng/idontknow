# 36-2 카피

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist     #CNN을 하는 사람들은 항상 mnist를 쓴다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, GlobalAveragePooling2D, LSTM
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) 
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train/255.
x_test = x_test/255. 


from tensorflow.keras.layers import Reshape

# print(x_train.shape)   #(60000, 28, 28)
# exit()

#2. 모델구성
model = Sequential()
model.add(Reshape(target_shape=(28,28,1), input_shape=(28,28)))
model.add(Conv2D(64, (3,3), input_shape=(28,28,1)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(16, (2,2), activation='relu'))
################################################
model.add(Reshape(target_shape=(23*23,16)))
model.add(LSTM(10,))
################################################
# model.add(GlobalAveragePooling2D()) 
model.add(Dense(units=32, activation='relu'))
model.add(Dense(10, activation='softmax'))
# model.summary()

# exit()
#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['acc'])

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
print('acc :', round(loss[1], 4))



# exit()
y_predict = model.predict(x_test)


y_predict = np.argmax(y_predict, axis=1)

# print(y_test.shape, y_predict.shape)
# exit()
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

sparse_categorical_crossentropy
loss : 0.11613986641168594
acc : 0.9639000296592712
accuracy_score :  0.9639
걸린시간 :  25.3 초

Reshape
1차
loss : 0.10235944390296936
acc : 0.97
accuracy_score :  0.9695
걸린시간 :  56.0 초

2차
loss : 0.09845181554555893
acc : 0.9706
accuracy_score :  0.9706
걸린시간 :  52.87 초

3차
loss : 0.06397169083356857
acc : 0.9807
accuracy_score :  0.9807
걸린시간 :  269.89 초


'''






