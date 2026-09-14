# keras31-3 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import time

#1. 데이터
path = "./_data/ddarung/" #재사용성을 위해 경로 분리
 
train_csv = pd.read_csv(path + "train.csv", index_col= 0) 
#index_col 이 컬럼은 인덱스니 제외하라고 알려주는 파라미터
print(train_csv) # [1459 * 10] 
print(train_csv.shape)
print(train_csv.columns)
print(train_csv.info())

# 결측치 처리 1. 삭제
train_csv = train_csv.dropna()
print(train_csv) # [1328 * 10]

#train_csv에서 x와 y를 분리
x = train_csv.drop(['count'], axis =1) # count 열(컬럼)을 지운다.
print("x: ", x) # [1328 * 9] 

y = train_csv['count'] 
print(y)
print(y.shape) # (1328,)

#id 컬럼을 제외하면 컬럼이 10개가 된다. id 컬럼은 y에게 영향을 주지 않는다.


test_csv = pd.read_csv(path + "test.csv", index_col= 0) #model.predict()에 넣을 값
print(test_csv) #[715 * 9] 
print(test_csv.shape)
print(test_csv.columns)
print(test_csv.info())


submission = pd.read_csv(path+"submission.csv" , index_col= 0) #여기에 model.predict(submission) 값을 넣는다.
print(submission) # NaN은 결측치를 의미, [715 * 9] 
print(submission.shape)
print(submission.columns)

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

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler             #preprocessing 이건 전처리다. OneHot에서 써봤다.
# scaler = MinMaxScaler()                                    #내가 받은 데이터가 무조건 쓰레기라고 상정을 해야된다. 내가 받은 데이터가 전처리가 안된 데이터가 대부분일꺼기 때문이다.
# scaler = StandardScaler()
scaler = MaxAbsScaler()                                      
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


print(np.min(x_train), np.max(x_train))      # 0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test))        # -0.0012367054167697258 1.0


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dropout(.2))
model.add(Dense(10))
model.add(Dropout(.3))
model.add(Dense(10))
model.add(Dropout(.5))
model.add(Dense(10))
model.add(Dense(1))

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
r2:  0.5520938141933568
mse:  2985.720382316494
RMSE :  54.64174578393789

dropout
r2:  0.5549884966119694
mse:  2966.424573123728
RMSE :  54.46489303325334
'''