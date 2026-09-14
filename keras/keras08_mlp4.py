from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array(range(10))
y = np.array([range(1,11), [10,9,8,7,6,5,4,3,2,1], [9,8,7,6,5,4,3,2,1,0]]).T

print('x.shape :', x.shape)
print('y.shape :', y.shape)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_shape = (1,)))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(3))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300, batch_size=1)


# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)
print('loss : %.8f' % loss)

result = model.predict(np.array([[10]]))
print('predict :', result)

# 하이퍼파라미터: 에폭, 레이어의 깊이, 노드 개수, 배치 사이즈