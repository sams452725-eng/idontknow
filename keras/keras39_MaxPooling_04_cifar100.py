import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint




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
model.add(Dense(100, activation='softmax')) 
# model.summary()

# exit()
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=12,
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

path = './_save/cifar100/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'      # 컴파일을 할때 hist에서 epoch와 val_loss가 있기에 거기서 땡겨온다.
# 04d 4자리의 int 표기, .4f는 소수점 4자리까지 표기 -----> {} Dictionary 방식
filepath = ''.join([path, 'cifar100_', date,'_', filename])      # ''은 파일명을 일단 공백으로 만든다. if, 'aa'가 있으면 aa 뒤로 join에 적어 놓은 것들이 들어온다.
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




# .4 이상

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


(padding & stride)
1차
loss : 2.2560784816741943
acc : 0.43720000982284546
accuracy_score :  0.4372
걸린시간 :  538.46 초

2차
loss : 2.285261631011963
acc : 0.4205999970436096
accuracy_score :  0.4206
걸린시간 :  314.05 초

3차
loss : 2.1911377906799316
acc : 0.4388999938964844
accuracy_score :  0.4389
걸린시간 :  577.79 초

4차
loss : 2.297391414642334
acc : 0.42879998683929443
accuracy_score :  0.4288
걸린시간 :  665.32 초

5차
loss : 2.2187793254852295
acc : 0.436599999666214
accuracy_score :  0.4366
걸린시간 :  876.83 초

6차
loss : 2.2275185585021973
acc : 0.43790000677108765
accuracy_score :  0.4379
걸린시간 :  774.62 초

7차
loss : 2.238840341567993
acc : 0.44670000672340393
accuracy_score :  0.4467
걸린시간 :  4342.83 초

8차(MAxPooling 적용)
loss : 1.974291443824768
acc : 0.49160000681877136
accuracy_score :  0.4916
걸린시간 :  653.13 초





'''