# 36-2 카피

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist     #CNN을 하는 사람들은 항상 mnist를 쓴다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
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
model.add(Conv2D(64, (5,5), input_shape=(28,28,1)))  
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))  
model.add(Dropout(.5))
model.add(Conv2D(32, (4,4), activation='relu'))  
model.add(Conv2D(16, (3,3), activation='relu'))  
model.add(Dropout(.4))
model.add(Conv2D(8, (2,2), activation='relu'))  
model.add(Dropout(.3))
model.add(Conv2D(4, (1,1), activation='relu'))  
model.add(Flatten())  #reshape와 동일한 기능으로 행 부분을 제외하고 열 부분을 2차원으로 만들어서 Dense와 바로 엮이게 만들어준다.
# model.summary()

# exit()
model.add(Dense(units=16, activation='relu'))  # units는 filters와 같이 output node의 개수를 일컫는 말이다.
model.add(Dropout(.2))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 마지막 Dense에서는 결국 (, 10)의 shape이 되어야 하는데, conv상태에서는 ( , , )형태이기 때문에 summary까지는 나오는데 이후 단계에서 error가 뜬다.



# exit()
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=3,
        restore_best_weights=True,
)


start_time=time.time()
model.fit(x_train, y_train, epochs=50, batch_size=512,
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
print('accuracy_score : ', acc_score, 3)
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

1차
loss : 0.06210283935070038
acc : 0.9894999861717224
accuracy_score :  0.9895 3
걸린시간 :  342.89 초

2차
loss : 0.028604781255126
acc : 0.9919999837875366
accuracy_score :  0.992 3
걸린시간 :  473.77 초

3차
loss : 0.03507482632994652
acc : 0.9921000003814697
ccuracy_score :  0.9921 3
걸린시간 :  465.88 초

4차
loss : 0.06381771713495255
acc : 0.9889000058174133
accuracy_score :  0.9889 3
걸린시간 :  471.38 초

5차
loss : 0.10985779017210007
acc : 0.9851999878883362
accuracy_score :  0.9852 3
걸린시간 :  463.53 초

6차
loss : 0.15381087362766266
acc : 0.9878000020980835
accuracy_score :  0.9878 3
걸린시간 :  572.84 초
'''






