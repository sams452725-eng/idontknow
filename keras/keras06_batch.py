from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=500, batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss :', loss)

# result = model.predict(np.array([1,2,3,4,5,6]))
# print('predict :', result)

# 하이퍼 파라미터 튜닝 요소: 노드 갯수, epoch 수, 레이어 깊이, 배치 사이즈