# keras33-5 카피

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error
import time

#1. 데이터
path = './_data/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)

print(train_csv)  #  [10886 rows x 11 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)

print(test_csv)  #  [6493 rows x 8 columns]

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)

print(submission)  #  [6493 rows x 1 columns]

print(train_csv.shape)
print(test_csv.shape)
print(submission.shape)



print(train_csv.info())
print(test_csv.info())
print(submission.info())

print(train_csv.describe())
############################### 결측치 확인 ###############################
print(train_csv.isna().sum())
print(train_csv.isnull().sum())


############################### x,y 분리 ###############################
x = train_csv.drop(['casual','registered','count'], axis = 1)
print(x)
y = train_csv['count']
print(y, y.shape)


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

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler              #preprocessing 이건 전처리다. OneHot에서 써봤다.
# scaler = MinMaxScaler()                                    #내가 받은 데이터가 무조건 쓰레기라고 상정을 해야된다. 내가 받은 데이터가 전처리가 안된 데이터가 대부분일꺼기 때문이다.
# scaler = StandardScaler()
scaler = MaxAbsScaler()                                      
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


print(np.min(x_train), np.max(x_train))      # 0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test))        # -0.0012367054167697258 1.0

#2. 모델 구성
# model = Sequential()
# model.add(Dense(10, input_dim=8))
# model.add(Dropout(.2))
# model.add(Dense(10))
# model.add(Dropout(.3))
# model.add(Dense(10))
# model.add(Dropout(.5))
# model.add(Dense(10))
# model.add(Dense(1))

input1 = Input(shape=(8,))     
dense1 = Dense(10,)(input1)      
drop1 = Dropout(.2)(dense1)
dense2 = Dense(10,)(drop1)         
drop2 = Dropout(.3)(dense2)
dense3 = Dense(10,)(drop2)         
drop3 = Dropout(.5)(dense3)
dense4 = Dense(10,)(drop3)         
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()







from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=30,
                   restore_best_weights=True,
                   verbose=1,
                   )



# mcp = ModelCheckpoint(
#     monitor='val_loss', 
#     mode='auto',
#     save_best_only=True,
#     filepath = filepath,             # filepath를 똑같이 써도되냐? 된다. why? 앞은 파라미터 뒤는 변수의 형식이다. 오히려 가독성을 위해서 동일하게 한다.
#     verbose=1,
# )


start_time = time.time()                
hist = model.fit(x_train, y_train,
                 epochs = 1000, batch_size = 32, validation_split=0.2,
                 callbacks=[es,],
                 verbose=1,
                 )
end_time = time.time()             



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
r2:  0.2708866596221924
mse:  24480.201171875
RMSE :  156.46150060597975

dropout
r2:  0.26593488454818726
mse:  24646.458984375
RMSE :  156.99190738498274

함수형
r2:  0.2632606625556946
mse:  24736.24609375
RMSE :  157.27760836733881
'''