from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1 . 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))


# 실습 8개, 4개, 4개 잘라봅시다!!!

# x_train = np.array(range(1,9))
x_train = x[:8]
# y_train = np.array(range(1,9))
y_train = y[:8]

# x_val = np.array([9,13])
x_val = x[8:12]
# y_val = np.array([9,13])
y_val = y[8:12]

# x_test = np.array([13,17])
x_test = x[12:]
# y_test = np.array([13,17])
y_test = y[12:]

print(x_train.shape, x_val.shape, x_test.shape)     #(8,) (4,) (4,)


# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=4,
            verbose=1,
            validation_data = (x_val, y_val),
            )




# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)


y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):           # RMSE함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))


rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)




y_pred = model.predict(x_test)
print('y_pred :', y_pred)

from sklearn.metrics import mean_absolute_error
print('MAE :', mean_absolute_error(y_test, y_pred))
