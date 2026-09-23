# 54-1 카피
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout

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


x = x.reshape(x.shape[0], x.shape[1], 1)   # (7, 3, 1)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(10, input_shape=(3, 1)))

model.add(SimpleRNN(units=10, input_length=3, input_dim=1))    # 위와 같은 형식이다.       

model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1))
model.summary()

exit()
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=3000)

#4. 평가, 예측
results = model.evaluate(x,y)
print('loss :', results)

x_predict = np.array([8,9,10]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[8,9,10]의 결과 :', y_predict)


































































































