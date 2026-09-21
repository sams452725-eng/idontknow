# 36-2 카피

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist     #CNN을 하는 사람들은 항상 mnist를 쓴다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
model = Sequential() 
model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(Conv2D(64, (3,3), activation='relu'))  
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))  
model.add(Conv2D(128, (3,3), activation='relu')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(256, (3,3), padding='same', activation='relu'))  
model.add(Conv2D(256, (3,3), activation='relu')) 
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Flatten())  
model.add(Dense(512, activation='relu'))     
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))     
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))  
# model.summary()

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
##################### mcp 세이브 파일명 만들기 start!! (mcp에 국한된게 아니라 일반적인 파일 생성시에 편하게 작업하는 방법) #####################
import datetime
date = datetime.datetime.now()    #현재 시간 반환
# print(date)   # 2026-09-14 11:42:51.252286
# print(type(date))   # <class 'datetime.datetime'>

date = date.strftime('%m%d_%H%M')
# print(date)    # 0914_1147
# print(type(date))   # <class 'str'>

path = './_save/mnist/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'      # 컴파일을 할때 hist에서 epoch와 val_loss가 있기에 거기서 땡겨온다.
# 04d 4자리의 int 표기, .4f는 소수점 4자리까지 표기 -----> {} Dictionary 방식
filepath = ''.join([path, 'mnist_', date,'_', filename])      # ''은 파일명을 일단 공백으로 만든다. if, 'aa'가 있으면 aa 뒤로 join에 적어 놓은 것들이 들어온다.
# 생성 파일명 ex) './_save/keras30/' + 'k30_' + '0914_1147'.keras

##################### mcp 세이브 파일명 만들기 finish!! #####################
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    save_best_only=True,
    filepath=filepath,
    verbose=1,
)
start_time=time.time()
hist = model.fit(x_train, y_train, epochs=150, batch_size=750,
          verbose=1,
          validation_split=.2,
          callbacks=[es,mcp]
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
1차
loss : 0.03257565200328827
acc : 0.9904999732971191
accuracy_score :  0.9905
걸린시간 :  64.17 초

2차
loss : 0.026502935215830803
acc : 0.992900013923645
accuracy_score :  0.9929
걸린시간 :  100.14 초

3차
loss : 0.025240128859877586
acc : 0.9926000237464905
accuracy_score :  0.9926
걸린시간 :  72.03 초

4차
loss : 0.027819735929369926
acc : 0.9929999709129333
accuracy_score :  0.993
걸린시간 :  108.71 초

5차
loss : 0.02462741546332836
acc : 0.9921000003814697
accuracy_score :  0.9921
걸린시간 :  206.17 초

6차
loss : 0.025240737944841385
acc : 0.992900013923645
accuracy_score :  0.9929
걸린시간 :  713.56 초

7차
loss : 0.02424255944788456
acc : 0.9925000071525574
accuracy_score :  0.9925
걸린시간 :  667.03 초

8차(MAxPooling 적용)
loss : 0.023152904585003853
acc : 0.9936000108718872
accuracy_score :  0.9936
걸린시간 :  412.29 초







'''






