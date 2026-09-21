
# 실습 레쓰기

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import pandas as pd
import os
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)
# print(np.max(x_train), np.min(x_train)) #255 0

###### 스케일링 1 ######
x_train = x_train/255. 
x_test = x_test/255. 
# print(np.max(x_train), np.min(x_train))  #1.0 0.0
# print(np.max(x_test), np.min(x_test))  #1.0 0.0

###### 스케일링 2 ######
# x_train = (x_train-127.5)/127.5        
# x_test = (x_test-127.5)/127.5
# print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
# print(np.max(x_test), np.min(x_test))   # 1.0 -1.0


x_train = x_train.reshape(-1,32,32,3)
x_test = x_test.reshape(-1,32,32,3)
# print(x_train.shape, x_test.shape)  #(150000, 32, 32, 1) (30000, 32, 32, 1)

# print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))


# exit()
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
# y_train = y_train.reshape(60000, 1)
y_train = y_train.reshape(-1, 1) 
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape, y_test.shape)  # (50000, 10) (10000, 10)


# exit()
#2. 모델구성
model = Sequential() 
model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(32,32,3))) 
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
        patience=14,
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

path = './_save/cifar10/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'      # 컴파일을 할때 hist에서 epoch와 val_loss가 있기에 거기서 땡겨온다.
# 04d 4자리의 int 표기, .4f는 소수점 4자리까지 표기 -----> {} Dictionary 방식
filepath = ''.join([path, 'cifar10_', date,'_', filename])      # ''은 파일명을 일단 공백으로 만든다. if, 'aa'가 있으면 aa 뒤로 join에 적어 놓은 것들이 들어온다.
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
hist = model.fit(x_train, y_train, epochs=100, batch_size=750,
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




# acc .67 이상

'''
1차
loss : 1.0224988460540771
acc : 0.6333000063896179
accuracy_score :  0.6333
걸린시간 :  163.68 초

2차
loss : 1.0427947044372559
acc : 0.6291000247001648
accuracy_score :  0.6291
걸린시간 :  160.96 초

3차
loss : 1.1954424381256104
acc : 0.597000002861023
accuracy_score :  0.597
걸린시간 :  186.3 초

4차
loss : 1.065340280532837
acc : 0.6323999762535095
accuracy_score :  0.6324
걸린시간 :  183.46 초

5차
loss : 1.008064866065979
acc : 0.6452999711036682
accuracy_score :  0.6453
걸린시간 :  229.86 초




(padding & stride)
1차
loss : 0.7159702181816101
acc : 0.7634999752044678
accuracy_score :  0.7635
걸린시간 :  442.44 초

2차
loss : 0.7642572522163391
acc : 0.7529000043869019
accuracy_score :  0.7529
걸린시간 :  367.02 초

3차
loss : 0.7255249619483948
acc : 0.7714999914169312
accuracy_score :  0.7715
걸린시간 :  160.99 초

4차
loss : 0.6830231547355652
acc : 0.7718999981880188
accuracy_score :  0.7719
걸린시간 :  466.42 초

5차(MAxPooling 적용)
loss : 0.5805103182792664
acc : 0.8084999918937683
accuracy_score :  0.8085
걸린시간 :  577.75 초





'''






