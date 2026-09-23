# keras23 카피

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras. layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping


############### acc >=.95 ###############


#1. 데이터
datasets = load_wine()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets['target']
# print(x.shape, y.shape)        #(178, 13) (178,)
# print(y)        #(178,)
# print(np.unique(y, return_counts=True))       #(array([0, 1, 2]), array([59, 71, 48]))
# print(pd.Categorical(y))


from tensorflow.keras.utils import to_categorical
y=to_categorical(y)
# print(y,y.shape)        # (178, 3)


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=.8,
    shuffle=True,
    random_state=338,
    stratify=y,
)

# print(x_train.shape, x_test.shape)          # (142, 13) (36, 13)
# print(y_train.shape, y_test.shape)          # (142, 3) (36, 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler             #preprocessing 이건 전처리다. OneHot에서 써봤다.
# scaler = MinMaxScaler()                                    #내가 받은 데이터가 무조건 쓰레기라고 상정을 해야된다. 내가 받은 데이터가 전처리가 안된 데이터가 대부분일꺼기 때문이다.
# scaler = StandardScaler()
scaler = MaxAbsScaler()                                      
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


print(np.min(x_train), np.max(x_train))      # 0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test))  


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))   


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
# learning_rate = 0.005
# learning_rate = 0.009
learning_rate = 0.01

from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping


model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'],)     
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=50,
    restore_best_weights=True,
)        
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5,        
)  
start_time = time.time()
model.fit(x_train, y_train, epochs=1500, batch_size=100,
          verbose=1,
          validation_split=.2,
          callbacks=[es],
          )
end_time = time.time()


#4. 평가,예측
result = model.evaluate(x_test, y_test,)
print('loss :', result[0])
print('acc :', round(result[1], 2))

y_predict = model.predict(x_test)
# print(y_predict)
y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)  
y_test = np.argmax(y_test, axis=1)
# print(y_test)     

# exit()
accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score :', accuracy_score)
# print('걸린시간 :', round(end_time - start_time, 2), '초')





'''
# (MinMaxScaler)
# loss : 0.01930960640311241
# acc : 1.0
# acc_score : 1.0

# (StandardScaler)
# loss : 0.01122838631272316
# acc : 1.0
# acc_score : 1.0

# # (MaxAbsScaler)                                      
# loss : 0.1360878348350525
# acc : 0.94
# acc_score : 0.9444444444444444






optimizer = learning_rate = 0.005
loss : 0.30360546708106995
acc : 0.92
acc_score : 0.9166666666666666

optimizer = learning_rate = 0.009
loss : 0.05238650366663933
acc : 0.97
acc_score : 0.9722222222222222

rlr
loss : 0.17899565398693085
acc : 0.97
acc_score : 0.9722222222222222










'''