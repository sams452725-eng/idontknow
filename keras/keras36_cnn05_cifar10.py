
# 실습 레쓰기
# acc .67 이상

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import pandas as pd
import os
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

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
model.add(Conv2D(128, (5,5), activation='relu', input_shape=(32,32,3))) 
model.add(Dropout(.2))
model.add(Conv2D(64, (4,4), activation='relu'))  
model.add(Conv2D(32, (3,3), activation='relu'))  
model.add(Conv2D(16, (2,2), activation='relu'))  
model.add(Dropout(.2))
model.add(Conv2D(8, (2,2), activation='relu'))  
model.add(Conv2D(4, (1,1), activation='relu'))  
model.add(Flatten())  
model.add(Dropout(.2))
model.add(Dense(64, activation='relu'))  
model.add(Dense(32, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(10, activation='softmax')) 
# model.summary()

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
start_time=time.time()
model.fit(x_train, y_train, epochs=100, batch_size=64,
          verbose=1,
          validation_split=.2,
          callbacks=[es]
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

6차
loss : 2.302614212036133
acc : 0.10000000149011612
accuracy_score :  0.1
걸린시간 :  195.99 초

7차
loss : 2.302598476409912
acc : 0.10000000149011612
accuracy_score :  0.1
걸린시간 :  222.17 초

8차
loss : 1.2163503170013428
acc : 0.5861999988555908
accuracy_score :  0.5862
걸린시간 :  149.74 초

9차
loss : 1.218441367149353
acc : 0.5712000131607056
accuracy_score :  0.5712
걸린시간 :  66.43 초

10차
loss : 1.1526050567626953
acc : 0.5985999703407288
accuracy_score :  0.5986
걸린시간 :  258.8 초


'''






