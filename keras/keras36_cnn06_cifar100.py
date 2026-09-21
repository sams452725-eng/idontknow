import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

# .4 이상



#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
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
# print(x_train.shape, x_test.shape)  #(50000, 32, 32, 3) (10000, 32, 32, 3)

# print(np.unique(y_train, return_counts=True))


# exit()
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
# y_train = y_train.reshape(60000, 1)
y_train = y_train.reshape(-1, 1) 
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape, y_test.shape)  # (50000, 100) (10000, 100)


# exit()
#2. 모델구성
model = Sequential() 
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(32,32,3))) 
# 30x30 -> 28x28
model.add(Conv2D(64, (3,3), activation='relu'))  
model.add(Dropout(0.2))
model.add(Conv2D(128, (5,5), activation='relu'))  
# 24x24 -> 20x20
model.add(Conv2D(128, (5,5), activation='relu'))  
model.add(Dropout(0.2))
model.add(Conv2D(256, (5,5), activation='relu'))  
# 16x16 -> 12x12
model.add(Conv2D(256, (5,5), activation='relu'))  
# 12x12 -> 8x8
model.add(Dropout(0.3))
model.add(Flatten())  # 8 x 8 x 256 = 16,384개 특징
model.add(Dense(512, activation='relu'))     # 100개 클래스를 위해 넉넉하게 512개
model.add(Dense(512, activation='relu'))     # 100개 클래스를 위해 넉넉하게 512개
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))     # 256개로 정제
model.add(Dense(256, activation='relu'))     # 256개로 정제
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))  # 최종 100개 출력
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
start_time=time.time()
model.fit(x_train, y_train, epochs=100, batch_size=755,
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
loss : 4.60521936416626
acc : 0.009999999776482582
accuracy_score :  0.01
걸린시간 :  38.93 초

2차
loss : 3.36574649810791
acc : 0.19609999656677246
accuracy_score :  0.1961
걸린시간 :  326.34 초

3차
loss : 4.1640825271606445
acc : 0.05460000038146973
accuracy_score :  0.0546
걸린시간 :  108.41 초

4차
loss : 4.60525369644165
acc : 0.009999999776482582
accuracy_score :  0.01
걸린시간 :  278.84 초

5차
loss : 3.0876405239105225
acc : 0.2558000087738037
accuracy_score :  0.2558
걸린시간 :  589.54 초

6차
loss : 2.9938650131225586
acc : 0.2689000070095062
accuracy_score :  0.2689
걸린시간 :  530.63 초

7차
loss : 2.798417806625366
acc : 0.3091999888420105
accuracy_score :  0.3092
걸린시간 :  609.45 초

8차
loss : 2.762537717819214
acc : 0.30489999055862427
accuracy_score :  0.3049
걸린시간 :  1519.24 초

9차
loss : 2.7217040061950684
acc : 0.32179999351501465
accuracy_score :  0.3218
걸린시간 :  777.13 초

10차
loss : 2.8382480144500732
acc : 0.30250000953674316
accuracy_score :  0.3025
걸린시간 :  573.76 초

11차
loss : 2.964949131011963
acc : 0.2921000123023987
accuracy_score :  0.2921
걸린시간 :  564.27 초

12차
loss : 2.8135721683502197
acc : 0.31850001215934753
accuracy_score :  0.3185
걸린시간 :  703.86 초



























'''