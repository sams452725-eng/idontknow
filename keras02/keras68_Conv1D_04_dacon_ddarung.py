# keras19-4 카피

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
path = 'C:/study/_data/ddarung/' #재사용성을 위해 경로 분리
 
train_csv = pd.read_csv(path + "train.csv", index_col= 0) 
#index_col 이 컬럼은 인덱스니 제외하라고 알려주는 파라미터
# print(train_csv) # [1459 * 10] 
# print(train_csv.shape)
# print(train_csv.columns)
# print(train_csv.info())

a = np.array(train_csv)
size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size +1) :
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)
bbb = split_x(a, size)

x = bbb[:, :-1]
y = bbb[:, -1]

# print(a.shape)


# exit()
# 결측치 처리 1. 삭제
train_csv = train_csv.dropna()
# print(train_csv) # [1328 * 10]

#train_csv에서 x와 y를 분리
x = train_csv.drop(['count'], axis =1) # count 열(컬럼)을 지운다.
# print("x: ", x) # [1328 * 9] 

y = train_csv['count'] 
# print(y)
# print(y.shape) # (1328,)

#id 컬럼을 제외하면 컬럼이 10개가 된다. id 컬럼은 y에게 영향을 주지 않는다.


test_csv = pd.read_csv(path + "test.csv", index_col= 0) #model.predict()에 넣을 값
# print(test_csv) #[715 * 9] 
# print(test_csv.shape)
# print(test_csv.columns)
# print(test_csv.info())


submission = pd.read_csv(path+"submission.csv" , index_col= 0) #여기에 model.predict(submission) 값을 넣는다.
# print(submission) # NaN은 결측치를 의미, [715 * 9] 
# print(submission.shape)
# print(submission.columns)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.75, 
    # test_size=0.25, 
    # shuffle=True,       # 디폴트 섞는다.
    random_state=4333,
)

# x_val, x_test, y_val, y_test = train_test_split(
#     x, y, 
#     train_size=0.75, 
#     # test_size=0.25, 
#     # shuffle=True,       # 디폴트 섞는다.
#     random_state=4333,
# )

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler             #preprocessing 이건 전처리다. OneHot에서 써봤다.
# scaler = MinMaxScaler()                                    #내가 받은 데이터가 무조건 쓰레기라고 상정을 해야된다. 내가 받은 데이터가 전처리가 안된 데이터가 대부분일꺼기 때문이다.
scaler = StandardScaler()
# scaler = MaxAbsScaler()                                      
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


# print(np.min(x_train), np.max(x_train))      # 0.0 1.0000000000000004
# print(np.min(x_test), np.max(x_test))        # -0.0012367054167697258 1.0

# print(x_train.shape, x_test.shape)    #(996, 9) (332, 9)

x_train = x_train.reshape(-1,9,1)
x_test = x_test.reshape(-1,9,1)

# print(x_train.shape, x_test.shape)    #(996, 9, 1) (332, 9, 1)

# exit()
#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, input_shape=(9,1), activation='relu', padding='same'))
model.add(Conv1D(32,3, padding='same', activation='relu'))
# model.add(MaxPooling1D())
model.add(Dropout(.2))
# model.add(Flatten())
# model.add(GlobalAveragePooling1D())
model.add(LSTM(32, return_sequences=True))      
model.add(LSTM(16))      
model.add(Dense(8, activation='relu'))
model.add(Dense(1))



#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.003
# learning_rate = 0.0001

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate), metrics=['mae'])
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=40,
        restore_best_weights=True,
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=15,
    verbose=1,
    factor=0.5,
)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 400, batch_size = 32,
            verbose=1,
            validation_split=0.2,
            callbacks=[es, rlr],
            )
end_time = time.time()


#4. 평가, 예측

print("========================================")
loss = model.evaluate(x_test, y_test)
print("========================================")
y_predict = model.predict(x_test)

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
r2:  0.5511961865757797
mse:  2991.7039234205395
RMSE :  54.69647084977731

(StandardScaler)
r2:  0.5395249491726799
mse:  3069.5038121149387
RMSE :  55.40310291053145

(MaxAbsScaler)                                      
r2:  0.5521828877851571
mse:  2985.126622178788
RMSE :  54.63631230398688

optimizer = learning_rate = 0.005
r2:  0.5511362171672199
mse:  2992.1036765186795
RMSE :  54.70012501373904

optimizer = learning_rate = 0.009
r2:  0.5574480711212644
mse:  2950.029171638028
RMSE :  54.314171002032495

Conv1D
1차
r2:  0.6981051629771249
mse:  2012.4159852626567
RMSE :  44.859959710889804

2차
r2:  -0.28095697185691004
mse:  8538.79553562278
RMSE :  92.40560337784056

3차
r2:  -0.1469728615839596
mse:  7645.6641129610225
RMSE :  87.43948829311059

4차
r2:  0.6852670297372829
mse:  2097.9943436326353
RMSE :  45.8038682169163










































'''