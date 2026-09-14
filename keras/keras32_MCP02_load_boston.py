# keras19-3 카피

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
import time

path = './_save/keras33/'


#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

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


from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler             #preprocessing 이건 전처리다. OneHot에서 써봤다.
# scaler = MinMaxScaler()                                    #내가 받은 데이터가 무조건 쓰레기라고 상정을 해야된다. 내가 받은 데이터가 전처리가 안된 데이터가 대부분일꺼기 때문이다.
# scaler = StandardScaler()
scaler = MaxAbsScaler()                                      
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


print(np.min(x_train), np.max(x_train))      # 0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test))        # -0.0012367054167697258 1.0



#2. 모델구성
# model = Sequential()
# model.add(Dense(10, input_dim=13))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(1))

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# es = EarlyStopping(monitor='val_loss', mode='min',
#                    patience=30,
#                    restore_best_weights=True,
#                    verbose=1,
#                    )

# ##################### mcp 세이브 파일명 만들기 start!! (mcp에 국한된게 아니라 일반적인 파일 생성시에 편하게 작업하는 방법) #####################
# import datetime
# date = datetime.datetime.now()    #현재 시간 반환
# # print(date)   # 2026-09-14 11:42:51.252286
# # print(type(date))   # <class 'datetime.datetime'>

# date = date.strftime('%m%d_%H%M')
# # print(date)    # 0914_1147
# # print(type(date))   # <class 'str'>

# path = './_save/keras33/'
# filename = '{epoch:04d}-{val_loss:.4f}.keras'      # 컴파일을 할때 hist에서 epoch와 val_loss가 있기에 거기서 땡겨온다.
# # 04d 4자리의 int 표기, .4f는 소수점 4자리까지 표기 -----> {} Dictionary 방식
# filepath = ''.join([path, 'k33_', date,'_', filename])      # ''은 파일명을 일단 공백으로 만든다. if, 'aa'가 있으면 aa 뒤로 join에 적어 놓은 것들이 들어온다.
# # 생성 파일명 ex) './_save/keras30/' + 'k30_' + '0914_1147'.keras

# ##################### mcp 세이브 파일명 만들기 finish!! #####################


# # exit()

# mcp = ModelCheckpoint(
#     monitor='val_loss', 
#     mode='auto',
#     save_best_only=True,
#     filepath = filepath,             # filepath를 똑같이 써도되냐? 된다. why? 앞은 파라미터 뒤는 변수의 형식이다. 오히려 가독성을 위해서 동일하게 한다.
#     verbose=1,
# )


# start_time = time.time()                
# hist = model.fit(x_train, y_train,
#                  epochs = 1000, batch_size = 32, validation_split=0.2,
#                  callbacks=[es,mcp],
#                  verbose=1,
#                  )
# end_time = time.time()             


model = load_model(path +'k33_0914_1401_0158-23.5405.keras')



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
r2:  0.7559285073383354
mse:  20.835562997717375
RMSE :  4.564598886837416







'''