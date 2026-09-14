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

# 0.0 1.0000000000000002

# exit()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.75, 
    # test_size=0.25, 
    # shuffle=True,       # 디폴트 섞는다.
    random_state=4333,
)

x_val, x_test, y_val, y_test = train_test_split(
    x, y, 
    train_size=0.75, 
    # test_size=0.25, 
    # shuffle=True,       # 디폴트 섞는다.
    random_state=4333,
)

from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler             #preprocessing 이건 전처리다. OneHot에서 써봤다.
from sklearn.preprocessing import RobustScaler
# scaler = MinMaxScaler()                                    # 내가 받은 데이터가 무조건 쓰레기라고 상정을 해야된다. 내가 받은 데이터가 전처리가 안된 데이터가 대부분일꺼기 때문이다.
# scaler = StandardScaler()                                    # 평균을 0으로 만들어서 좌우로 퍼진 데이터의 값을 좀 줄인다.
# scaler = MaxAbsScaler()                                      # 최대의 절대값(Abs)을 1로 만든다.
scaler = RobustScaler()               # RobustScaler는 이상치에 강력하다. 즉, RobustScaler을 사용했을 때 결과가 잘 나왔다면 이상치가 많이 조정됐다는 의미다.

# 모든 모델은 만들어져서 배포할때까지 여러번 돌린다. 단 한번으로 모델을 만들어서 배포 할수는 없다.

'''
scaler.fit(x_train)
x_train = scaler.transform(x_train)        이렇게 fit 따로 transform 따로 써도 되지만 이러면 2줄로써 가독성이 좋지않다. 때문에 아래 방식으로 해도 된다.
'''
x_train = scaler.fit_transform(x_train)
# x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


print(np.min(x_train), np.max(x_train))      # 0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test))        # -0.0012367054167697258 1.0





# 잘 모를땐 MinMaxScaler를 fit 할때는 x_train만 fit 한다

# exit()
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








'''
(MinMaxScaler)
r2:  0.5941637950083452
mse:  0.5272858127524316
RMSE :  0.7261444847634881

(StandardScaler)
r2:  0.5974904058969365
mse:  0.5229636879529992
RMSE :  0.7231622832760287

(MaxAbsScaler)
r2:  0.5643445731151149
mse:  0.5660286662934072
RMSE :  0.7523487663932248

(RobustScaler)
r2:  0.5971821626846103
mse:  0.5233641753189308
RMSE :  0.7234391303481799















'''