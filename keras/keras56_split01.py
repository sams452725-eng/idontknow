import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import time

a = np.array(range(1,11))
# print(a)       #[ 1  2  3  4  5  6  7  8  9 10]
# exit()
size = 5        # timestep 사이즈

# print(a.shape)  #(10,)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size +1) :
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
# print(bbb, bbb.shape)      # (6, 5)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
x = bbb[:, :-1]
y = bbb[:, -1]

# print(x, y, x.shape, y.shape)     # (6, 4) (6,)

x = x.reshape(x.shape[0], x.shape[1], 1)   #(6, 4, 1)

#2. 모델구성
model = Sequential()
# model.add(GRU(16,input_shape=(4,1)))            
model.add(LSTM(16, input_shape=(4, 1)))        
# model.add(SimpleRNN(16, input_shape=(4, 1)))        
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

#4. 평가, 예측n
results = model.evaluate(x,y)
print('loss :', results)

x_predict = np.array([7,8,9,10]).reshape(1,4,1)
y_predict = model.predict(x_predict)

print('[7,8,9,10]의 결과 :', y_predict)  


'''
[7,8,9,10]의 결과 : [[10.744997]]









'''