# keras55-2 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, Bidirectional, LSTM, GRU

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])       

# print(x.shape, y.shape)       # (13, 3) (13,)

x = x.reshape(x.shape[0], x.shape[1], 1)   # (13, 3, 1)


#2. 모델구성
model = Sequential()
# model.add(Bidirectional(GRU(10), input_shape=(3, 1)))
# model.add(Bidirectional(LSTM(10), input_shape=(3, 1)))
model.add(Bidirectional(SimpleRNN(10), input_shape=(3, 1)))   
# model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
# model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=3000)

#4. 평가, 예측n
results = model.evaluate(x,y)
print('loss :', results)

x_predict = np.array([50,60,70]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[50,60,70]의 결과 :', y_predict)       # 80에 맞춰본다.


'''
[50,60,70]의 결과 : [[78.10299]]
[50,60,70]의 결과 : [[75.91455]]
[50,60,70]의 결과 : [[77.29082]]
[50,60,70]의 결과 : [[75.42901]]
[50,60,70]의 결과 : [[78.669044]]
[50,60,70]의 결과 : [[78.0828]]
[50,60,70]의 결과 : [[79.15907]]
[50,60,70]의 결과 : [[79.272316]]
[50,60,70]의 결과 : [[79.30197]]

Bidirectional 적용
[50,60,70]의 결과 : [[77.67835]]
[50,60,70]의 결과 : [[77.0766]]
[50,60,70]의 결과 : [[77.00341]]
[50,60,70]의 결과 : [[75.99113]]
[50,60,70]의 결과 : [[77.59326]]
[50,60,70]의 결과 : [[78.06903]]
[50,60,70]의 결과 : [[78.49164]]
[50,60,70]의 결과 : [[76.74505]]








'''