# keras28-2 카피


from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
import time
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, Flatten, Conv2D, Reshape,  Conv1D, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target



x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.75, 
    # test_size=0.25, 
    # shuffle=True,       # 디폴트 섞는다.
    random_state=4333,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler           #preprocessing 이건 전처리다. OneHot에서 써봤다.
from sklearn.preprocessing import RobustScaler
# scaler = MinMaxScaler()                                    #내가 받은 데이터가 무조건 쓰레기라고 상정을 해야된다. 내가 받은 데이터가 전처리가 안된 데이터가 대부분일꺼기 때문이다.
# scaler = StandardScaler()
# scaler = MaxAbsScaler()                                      
scaler = RobustScaler()               
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


# print(np.min(x_train), np.max(x_train))      # 0.0 1.0000000000000004
# print(np.min(x_test), np.max(x_test))        # -0.0012367054167697258 1.0


# print(x_train.shape, x_test.shape)    #(331, 10) (111, 10)

x_train = x_train.reshape(-1,10,1)
x_test = x_test.reshape(-1,10,1)

# print(x_train.shape, x_test.shape)    #(331, 10, 1) (111, 10, 1)

# exit()
#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, input_shape=(10,1), activation='relu', padding='same'))
model.add(Conv1D(32,3, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(Dropout(.2))
# model.add(Flatten())
# model.add(GlobalAveragePooling1D())
model.add(LSTM(32, return_sequences=True))      
model.add(LSTM(16))      
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.005
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate), metrics=['mae'])
start_time = time.time()           #현재시간을 반환. 즉, 시작 시간     
hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 32,
            verbose=1,
            # validation_data = (x_val, y_val),
            )
end_time = time.time()             #현재시간을 반환. 즉, 끝 시간

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss(MSE) :', loss)
print('MAE       :', loss[1])

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)

def RMSE(y_test, y_predict):  #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("Test R2   :", r2)
print("Test RMSE :", rmse)
print('걸린시간1 :', round(end_time - start_time, 2), '초')


'''
# (MinMaxScaler)
# r2:  0.43784708911300696
# mse:  3174.7342229813144
# RMSE :  56.34477990889054

# (StandardScaler)
# r2:  0.4401149734996066
# mse:  3161.926444107515
# RMSE :  56.23100963087463

# (MaxAbsScaler)
# r2:  0.4407357155077316
# mse:  3158.4208304947933
# RMSE :  56.19982945254188

scaler = RobustScaler()      
r2:  0.44519795632075054
mse:  3133.2205187185496
RMSE :  55.97517770153615


optimizer = learning_rate = 0.005
r2:  0.4409794304347616
mse:  3157.044460997262
RMSE :  56.1875828008045



optimizer = learning_rate = 0.009
r2:  0.4433164352013548
mse:  3143.846327770366
RMSE :  56.070012732033206






loss(MSE) : [4316.56005859375, 54.16385269165039]
MAE       : 54.16385269165039
mse:  4316.5600223856045
Test R2   : 0.2356630158086862
Test RMSE : 65.70053289270646
걸린시간1 : 44.38 초







'''