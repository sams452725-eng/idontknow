from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler

# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape)  #(404, 13)
print(x_test.shape)   #(102, 13)
print(y_train.shape)  #(404,)
print(y_test.shape)   #(102,)

# 데이터 정규화 (StandardScaler)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train[0])
print(x_test[0])

# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 13))

# 3. 컴파일
model.compile(loss='mse', optimizer='adam')

# 4. 훈련
model.fit(x_train, y_train, epochs=1000, batch_size=2, validation_split=0.2)

# 5. 평가
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

# 6. 예측
y_pred = model.predict(x_test)
print('y_pred :', y_pred)

from sklearn.metrics import mean_absolute_error
print('MAE :', mean_absolute_error(y_test, y_pred))