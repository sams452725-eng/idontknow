# 실습

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist     
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Input
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
'''
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
model.add(GlobalAveragePooling2D()) 

model.add(Dense(1024, activation='relu'))     
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))     
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))     
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))     
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))     
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))     
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))     
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))   
# model.summary()
'''
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
        patience=13,
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

path = './_save/fashion/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'      # 컴파일을 할때 hist에서 epoch와 val_loss가 있기에 거기서 땡겨온다.
# 04d 4자리의 int 표기, .4f는 소수점 4자리까지 표기 -----> {} Dictionary 방식
filepath = ''.join([path, 'fashion_', date,'_', filename])      # ''은 파일명을 일단 공백으로 만든다. if, 'aa'가 있으면 aa 뒤로 join에 적어 놓은 것들이 들어온다.
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




# acc 기준 .93 이상

''''
loss : 0.3461201786994934
acc : 0.9070000052452087
accuracy_score :  0.907
걸린시간 :  119.28 초



(padding & stride)
1차
loss : 0.4317833483219147
acc : 0.9269999861717224
accuracy_score :  0.927
걸린시간 :  721.59 초

2차
loss : 0.24379892647266388
acc : 0.9182000160217285
accuracy_score :  0.9182
걸린시간 :  361.78 초

3차
loss : 0.2446223944425583
acc : 0.9157000184059143
accuracy_score :  0.9157
걸린시간 :  257.37 초

4차
loss : 0.2396046668291092
acc : 0.9168000221252441
accuracy_score :  0.9168
걸린시간 :  1512.87 초

5차
loss : 0.23752163350582123
acc : 0.9190000295639038
accuracy_score :  0.919
걸린시간 :  639.47 초

6차
loss : 0.23025421798229218
acc : 0.9162999987602234
accuracy_score :  0.9163
걸린시간 :  665.29 초

7차
loss : 0.23977607488632202
acc : 0.9169999957084656
accuracy_score :  0.917
걸린시간 :  84.67 초

8차(MAxPooling 적용)
loss : 0.2141590267419815
acc : 0.9265999794006348
accuracy_score :  0.9266
걸린시간 :  381.36 초

9차(GAP 적용)
loss : 0.21042278409004211
acc : 0.9265000224113464
accuracy_score :  0.9265
걸린시간 :  493.16 초

10차
loss : 0.278239369392395
acc : 0.9236000180244446
accuracy_score :  0.9236
걸린시간 :  681.83 초

11차
loss : 0.22309304773807526
acc : 0.9223999977111816
accuracy_score :  0.9224
걸린시간 :  141.24 초

















'''








