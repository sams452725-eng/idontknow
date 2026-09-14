# 19-1카피


# import
# ssl._create_default_https_context = ssl._create_unverified_context


from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import time


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  #(20640, 8) (20640,)

'''
MinMaxScaler(camelcase)의 계산식

원값 - Min
----------
Max - Min

'''
from sklearn.preprocessing import MinMaxScaler             #preprocessing 이건 전처리다. 전처리 식은 OneHot에서 써봤다.
scaler = MinMaxScaler()                                    #내가 받은 데이터가 무조건 쓰레기라고 상정을 해야된다. 내가 받은 데이터가 전처리가 안된 데이터가 대부분일꺼기 때문이다.
scaler.fit(x)
x = scaler.transform(x)

print(x)
print(np.min(x), np.max(x))     # 0.0 1.0000000000000002




# exit()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.75, 
    # test_size=0.25, 
    # shuffle=True,       # 디폴트 섞는다.
    random_state=4333,
)




#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()           #현재시간을 반환. 즉, 시작 시간     
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32,
            verbose=1,
            )
end_time = time.time()             #현재시간을 반환. 즉, 끝 시간

#4. 평가, 예측
print("========================================")
loss = model.evaluate(x_test, y_test, batch_size = 18)
print("========================================")
y_predict = model.predict(x_test, batch_size = 18)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)

def RMSE(y_test, y_predict):  #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# r2:  0.5865100124050097
# mse:  0.537230048705242
# # RMSE :  0.7329597865539705