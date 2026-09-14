from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array(range(1, 11))
y = np.array(range(1, 11))

# x_train = np.array(range(1,8))
# y_train = np.array(range(1,8))

# x_test = np.array(range(8,11))
# y_test = np.array(range(8,11))

# numpy list의 슬라이싱을 이용해 7:3 으로 나누자
x_train = x[:7]
y_train = y[:7]

x_test = x[7:]
y_test = y[7:]

print('x_train.shape :', x_train.shape)
print('y_train.shape :', y_train.shape)
print('x_test.shape :', x_test.shape)
print('y_test.shape :', y_test.shape)

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