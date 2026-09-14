from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array([[1,2,3,4,5], [6,7,8,9,10]])
x = x.T # x = x.transpose()
# x = np.array([[1,6], [2,7], [3,8], [4,9], [5,10]])
y = np.array([1,2,3,4,5])

print('x.shape :', x.shape) # (5, 2)
print('y.shape :', y.shape) # (5,)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_shape = (2,)))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)


# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)

# result = model.predict(np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]]))
result = model.predict(np.array([[6, 11]]))
print('predict :', result)

# 하이퍼파라미터: 에폭, 레이어의 깊이, 노드 개수, 배치 사이즈