import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import time

#1. 데이터
a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0]]).T

size=5


# exit()
# print(a.shape)      #(10, 2)

# split_x를 이용해서
# 칠판처럼 짤라 보거라!!!
# 이녀석아~~!!!

def split_x(dataset, size):           # 벡터 형태의 데이터 뿐만 아니라 매트릭스 형태의 데이터까지 알아서 짤라준다.
    aaa = []
    for i in range(len(dataset) - size +1) :
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
# print(bbb, bbb.shape)    # (6, 5, 2)
# [[[ 1  9]
#   [ 2  8]
#   [ 3  7]
#   [ 4  6]
#   [ 5  5]]

#  [[ 2  8]
#   [ 3  7]
#   [ 4  6]
#   [ 5  5]
#   [ 6  4]]

#  [[ 3  7]
#   [ 4  6]
#   [ 5  5]
#   [ 6  4]
#   [ 7  3]]

#  [[ 4  6]
#   [ 5  5]
#   [ 6  4]
#   [ 7  3]
#   [ 8  2]]

#  [[ 5  5]
#   [ 6  4]
#   [ 7  3]
#   [ 8  2]
#   [ 9  1]]

#  [[ 6  4]
#   [ 7  3]
#   [ 8  2]
#   [ 9  1]
#   [10  0]]] 

# x = bbb[:, :-1, :]
x = bbb[:, :-1,]

# y = bbb[:, -1, 1]
y = bbb[:, -1, -1]

# print(x, y, x.shape, y.shape)      # (6, 4, 2) (6,)
# exit()
#2. 모델구성
model = Sequential()
# model.add(GRU(16,input_shape=(4,2)))            
model.add(LSTM(16, input_shape=(4, 2)))        
# model.add(SimpleRNN(16, input_shape=(4, 2)))        
# model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=3000)

#4. 평가, 예측
results = model.evaluate(x,y)
print('loss :', results)

x_predict = np.array([[7,8,9,10],[3,2,1,0]]).T.reshape(1,4,2)
y_predict = model.predict(x_predict)

print('[[7,8,9,10],[3,2,1,0]].T의 결과 :', y_predict[-1])  


'''

















'''