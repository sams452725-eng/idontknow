from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array(range(1, 11))
y = np.array(range(1, 11))

x_train = np.array(range(1,8))
y_train = np.array(range(1,8))

x_test = np.array(range(8,11))
y_test = np.array(range(8,11))

print('x.shape :', x.shape)
print('y.shape :', y.shape)

# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
print('loss : %.8f' % loss)

result = model.predict(np.array([[10]]))
print('predict :', result)

# 하이퍼파라미터: 에폭, 레이어의 깊이, 노드 개수, 배치 사이즈