from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. 데이터
x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,38,12,13,8,19,22,52,11,18,17,42,25,21])

# x_train = np.array(range(1,8))
# y_train = np.array(range(1,8))

# x_test = np.array(range(8,11))
# y_test = np.array(range(8,11))

# 사이킷런을 이용해 랜덤하게 7:3 으로 나누자
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)

print('x_train.shape :', x_train.shape)
print('y_train.shape :', y_train.shape)
print('x_test.shape :', x_test.shape)
print('y_test.shape :', y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1)

print("===================================================")
# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
print('loss : %.8f' % loss)

result = model.predict(x)
print('predict :', result)

# 하이퍼파라미터: 에폭, 레이어의 깊이, 노드 개수, 배치 사이즈

plt.scatter(x, y)
plt.plot(x, result, color='red')
plt.show()