# 27-1카피


# import
# ssl._create_default_https_context = ssl._create_unverified_context


from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import time
from tensorflow.keras.callbacks import EarlyStopping


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape)  #(20640, 8) (20640,)

from sklearn.preprocessing import MinMaxScaler             #preprocessing 이건 전처리다. 전처리 식은 OneHot에서 써봤다.
scaler = MinMaxScaler()                                    #내가 받은 데이터가 무조건 쓰레기라고 상정을 해야된다. 내가 받은 데이터가 전처리가 안된 데이터가 대부분일꺼기 때문이다.
scaler.fit(x)
x = scaler.transform(x)

# print(x)
# print(np.min(x), np.max(x))     # 0.0 1.0000000000000002


x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.75, 
    # test_size=0.25, 
    # shuffle=True,       # 디폴트 섞는다.
    random_state=4333,
)




#2. 모델구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=8))
model.add(Dense(10))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001          #디폴트
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

from tensorflow.keras.callbacks import ReduceLROnPlateau

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=40,
    verbose=1,
    restore_best_weights=True,
)          
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5,        #LeraningRate를 0.5를 곱한만큼 줄어든다./ Adam은 자체 LR조절 알고리즘이 있지만 ReduceLROnPlateau와는 다른 알고리즘이기 때문에 상호보완적이다.
)


start_time = time.time()           #현재시간을 반환. 즉, 시작 시간     
hist = model.fit(x_train, y_train,
                 callbacks=[es, rlr],
                 epochs = 500, batch_size = 32,
                 verbose=1, validation_split=0.2,
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
optimizer = 'adam'
r2:  0.5865100124050097
mse:  0.537230048705242
# RMSE :  0.7329597865539705



optimizer = learning_rate = 0.01
r2:  0.7315907980957288
mse:  0.3487327213185315
RMSE :  0.590535961071408

optimizer = learning_rate = 0.0001
r2:  0.635924388415503
mse:  0.47302804036820123
RMSE :  0.6877703398433239

optimizer = learning_rate = 0.005
r2:  0.7552160234112568
mse:  0.3180374655016836
RMSE :  0.5639481053267965

optimizer = learning_rate = 0.05
r2:  0.6743952828081168
mse:  0.4230444347469643
RMSE :  0.6504186611306323

optimizer = learning_rate = 0.009
r2:  0.7605395285166753
mse:  0.3111208605224408
RMSE :  0.5577820905357582

rlr
r2:  0.730948711487377
mse:  0.3495669572860917
RMSE :  0.5912418771417428














'''