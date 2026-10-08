# keras54-1 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from tensorflow.keras.layers import Conv1D, Flatten, GlobalAveragePooling1D
#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],       
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ])
y = np.array([4,5,6,7,8,9,10])

# print(x.shape, y.shape)       # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)   # (7, 3, 1)------> 시계열 데이터기 때문에 각각의 feature 형태이기 때문에 reshape가 들어갔다.
# print(x.shape)   # (7, 3, 1)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(3,1)))   #Conv2D랑 똑같다!! 3차원으로 들어갔다 3차원으로 나온다.
model.add(Conv1D(10,2))
model.add(GlobalAveragePooling1D())
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
# model.summary()

# exit()
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=3000)

#4. 평가, 예측
results = model.evaluate(x,y)
print('loss :', results)

x_predict = np.array([8,9,10]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[8,9,10]의 결과 :', y_predict)
'''
[8,9,10]의 결과 : [[10.430394]]
[8,9,10]의 결과 : [[10.80435]]
[8,9,10]의 결과 : [[11.238563]]

[8,9,10]의 결과 : [[11.000002]]

'''
































































































