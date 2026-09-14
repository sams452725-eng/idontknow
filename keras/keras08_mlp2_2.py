from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array(range(10)) # [0,1,2,3,4,5,6,7,8,9]
y = np.array(range(1,11)) # [1,2,3,4,5,6,7,8,9,10]

x = np.array([range(10), range(21, 31), range(201, 211)]).T
y = np.array(range(1,11))

print('x.shape :', x.shape) # (10, 3)
print('y.shape :', y.shape) # (10,)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_shape = (3,)))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=800, batch_size=1)


# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)
print('loss : %.8f' % loss)

result = model.predict(np.array([[10, 31, 211]]))
print('predict :', result)

# 하이퍼파라미터: 에폭, 레이어의 깊이, 노드 개수, 배치 사이즈