from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape) #(20640, 8)
print(y.shape) #(20640,)

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 데이터 정규화 (StandardScaler)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train[0])
print(x_test[0])

# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 8))

# 3. 컴파일
model.compile(loss='mse', optimizer='adam')

# 4. 훈련
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

# 5. 평가
loss = model.evaluate(x_test, y_test)
print('loss :', loss)