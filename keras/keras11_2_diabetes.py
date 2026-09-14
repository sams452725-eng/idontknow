from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape) #(442, 10)
print(y.shape) #(442,)

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
model.add(Dense(1, input_dim = 10))

# 3. 컴파일
model.compile(loss='mse', optimizer='adam')

# 4. 훈련
model.fit(x_train, y_train, epochs=1000, batch_size=4, validation_split=0.2)

# 5. 평가
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

# 6. 예측
y_pred = model.predict(x_test)
print('y_pred :', y_pred)

from sklearn.metrics import mean_absolute_error
print('MAE :', mean_absolute_error(y_test, y_pred))