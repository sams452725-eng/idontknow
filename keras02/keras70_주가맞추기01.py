# 삼성전자 내일 아침 시가 맞추기
# 앙상블 쓸것
# LSTM, Conv1D 쓸것
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten, MaxPooling1D, GlobalAveragePooling1D, Input, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import concatenate, Concatenate


#1. 데이터
path = './_data/samsung/'
datasets1 = pd.read_csv(path + '삼성전자261012.csv', index_col=0, encoding='cp949', thousands=',')
datasets2 = pd.read_csv(path + '아모레261012.csv', index_col=0, encoding='cp949', thousands=',')
datasets3 = pd.read_csv(path + '하이닉스261012.csv', index_col=0, encoding='cp949', thousands=',')
# print(datasets1.shape,datasets2.shape,datasets3.shape)    # (1120, 16) (2060, 16) (1300, 16)

datasets2 = datasets2[:1120]
datasets3 = datasets3[:1120]

datasets1 = datasets1.drop(['대비'], axis=1)
datasets2 = datasets2.drop(['대비'], axis=1)
datasets3 = datasets3.drop(['대비'], axis=1)

datasets1 = datasets1[::-1]
datasets2 = datasets2[::-1]
datasets3 = datasets3[::-1]

x1_data = datasets1.drop(['시가'], axis=1)    
x2_data = datasets2    
x3_data = datasets3    

y_data = datasets1['시가']

# print(x1_data.shape,x2_data.shape,x3_data.shape)    # (1120, 14) (1120, 15) (1120, 15)

scaler = MinMaxScaler()

x1_data = scaler.fit_transform(x1_data)
x2_data = scaler.fit_transform(x2_data)
x3_data = scaler.fit_transform(x3_data)

y_raw = y_data.values.reshape(-1, 1)
y_scaled = scaler.fit_transform(y_raw)


def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size +1) :
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

size = 20

x1 = split_x(x1_data[:-1], size)
x2 = split_x(x2_data[:-1], size)
x3 = split_x(x3_data[:-1], size)

y = y_scaled[size:]

x1_pred = x1_data[-size:].reshape(1, size, x1_data.shape[1])
x2_pred = x2_data[-size:].reshape(1, size, x2_data.shape[1])
x3_pred = x3_data[-size:].reshape(1, size, x3_data.shape[1])


x1_train, x1_test, y_train, y_test = train_test_split(x1, y, train_size=0.8, shuffle=True, random_state=4333)
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, train_size=0.8, shuffle=True, random_state=4333)
x3_train, x3_test, y_train, y_test = train_test_split(x3, y, train_size=0.8, shuffle=True, random_state=4333)


# print(x1_train.shape,x1_test.shape)    # (880, 20, 14) (220, 20, 14)
# print(x2_train.shape,x2_test.shape)    # (880, 20, 15) (220, 20, 15)
# print(x3_train.shape,x3_test.shape)    # (880, 20, 15) (220, 20, 15)


#2-1. 모델
input1 = Input(shape=(20,14))
dense1 = LSTM(8, return_sequences=True)(input1)
dense2 = Bidirectional(LSTM(16, return_sequences=True))(dense1)
dense3 = Conv1D(32, 2, activation='relu', padding='same')(dense2)
dense4 = MaxPooling1D(5)(dense3)
dense5 = Flatten()(dense4)
output1 = Dense(32, activation='relu')(dense5)
# model1 = Model(inputs=input1, outputs=output1)

#2-2. 모델
input21 = Input(shape=(20,15))
dense21 = LSTM(64, return_sequences=True)(input21)
dense22 = Bidirectional(LSTM(32, return_sequences=True))(dense21)
dense23 = Bidirectional(LSTM(32, return_sequences=True))(dense22)
dense24 = Conv1D(16, 2, activation='relu', padding='same')(dense23)
dense25 = MaxPooling1D(3)(dense24)
dense26 = Flatten()(dense25)
output21 = Dense(4, activation='relu')(dense26)
# model2 = Model(inputs=input21, outputs=output21)

#2-3. 모델
input31 = Input(shape=(20,15))
dense31 = LSTM(32, return_sequences=True)(input31)
dense32 = Bidirectional(LSTM(16, return_sequences=True))(dense31)
dense33 = Bidirectional(LSTM(16, return_sequences=True))(dense32)
dense34 = Conv1D(8, 2, activation='relu', padding='same')(dense33)
dense35 = MaxPooling1D(4)(dense34)
dense36 = Flatten()(dense35)
output31 = Dense(32, activation='relu')(dense36)

#2-4. 모델 합치기
merge1 = Concatenate()([output1, output21, output31])
merge2 = Dense(16)(merge1)
last_output = Dense(1)(merge2)

model = Model(inputs=[input1, input21, input31], outputs=last_output)
# model.summary()



#3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam
learning_rate = 0.005

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate), metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=15,
        restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=10,
    verbose=1,
    factor=0.5,        
)  
import datetime
date = datetime.datetime.now()    

date = date.strftime('%m%d_%H%M')

path = './_save/keras70/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'     
filepath = ''.join([path, 'keras70', date,'_', filename])


mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='min',
    save_best_only=True,
    filepath = filepath,             
    verbose=1,
)

start_time2 = time.time()                      
model.fit([x1_train, x2_train, x3_train], y_train, 
        epochs = 3000,
        batch_size = 32,
        callbacks=[es,mcp,rlr],
        validation_split=0.2,

)
end_time2 = time.time()

# exit()


#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)
print('loss(MSE) :', loss[0])
print('MAE       :', loss[1])

y_predict = model.predict([x1_test, x2_test, x3_test])

y_test_before_scale = scaler.inverse_transform(y_test)
y_predict_before_scale = scaler.inverse_transform(y_predict)
mse = mean_squared_error(y_test_before_scale, y_predict_before_scale)

def RMSE(y_test, y_predict):  #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test_before_scale, y_predict_before_scale)

r2 = r2_score(y_test_before_scale, y_predict_before_scale)

print("Test R2   :", r2)
print("Test RMSE :", rmse, "원")
print('걸린시간  :', round(end_time2 - start_time2, 2), '초')

y_pred_result = model.predict([x1_pred, x2_pred, x3_pred])
tomorrow_price = scaler.inverse_transform(y_pred_result)

print("=================================================")
print("★ 2026-10-13 삼성전자 시가 예측값 :", tomorrow_price[0][0], "원 ★")
print("=================================================")



'''
★ 최고 성적: 9차시 ★
loss(MSE) : 0.00011265902867307886
MAE       : 0.005798144266009331
Test R2   : 0.9968995005123993
Test RMSE : 3420.923169424128 원
걸린시간  : 65.9 초
★ 2026-10-13 삼성전자 시가 예측값 : 269535.3 원 ★
모델 가중치 파일 : keras701012_1615_0138-0.0001.keras

==================================================
전체 실험 기록 (1차 ~ 11차)
==================================================
1차
loss(MSE) : 0.00016250277985818684
MAE       : 0.005617406219244003
Test R2   : 0.9955277479885942
Test RMSE : 4108.569355306832 원
걸린시간  : 72.19 초
★ 2026-10-13 삼성전자 시가 예측값 : 267305.75 원 ★
keras701012_1523_0070-0.0001.keras

2차
loss(MSE) : 0.00015677047485951334
MAE       : 0.00685152830556035
Test R2   : 0.9956855047275419
Test RMSE : 4035.4547997922023 원
걸린시간  : 108.87 초
★ 2026-10-13 삼성전자 시가 예측값 : 264593.44 원 ★
keras701012_1526_0097-0.0001.keras

3차
loss(MSE) : 0.00019632546172942966
MAE       : 0.008922778069972992
Test R2   : 0.9945969100356468
Test RMSE : 4515.944417138484 원
걸린시간  : 76.76 초
★ 2026-10-13 삼성전자 시가 예측값 : 241485.1 원 ★
keras701012_1530_0077-0.0002.keras

4차
loss(MSE) : 0.0001527714339317754
MAE       : 0.00609528599306941
Test R2   : 0.9957955622790665
Test RMSE : 3983.652529024624 원
걸린시간  : 114.57 초
★ 2026-10-13 삼성전자 시가 예측값 : 263150.6 원 ★
keras701012_1536_0132-0.0001.keras

5차
loss(MSE) : 0.00017442391254007816
MAE       : 0.008004914969205856
Test R2   : 0.99519966441901
Test RMSE : 4256.604384295374 원
걸린시간  : 33.6 초
★ 2026-10-13 삼성전자 시가 예측값 : 271241.97 원 ★
keras701012_1607_0087-0.0002.keras

6차
loss(MSE) : 0.0001543086691526696
MAE       : 0.007005053572356701
Test R2   : 0.9957532574780051
Test RMSE : 4003.644003464357 원
걸린시간  : 62.11 초
★ 2026-10-13 삼성전자 시가 예측값 : 259957.28 원 ★
keras701012_1610_0195-0.0002.keras

7차
loss(MSE) : 0.00017165954341180623
MAE       : 0.006943927146494389
Test R2   : 0.9952757425679291
Test RMSE : 4222.739259701195 원
걸린시간  : 36.67 초
★ 2026-10-13 삼성전자 시가 예측값 : 266597.44 원 ★
keras701012_1612_0082-0.0001.keras

8차
loss(MSE) : 0.0001415276201441884
MAE       : 0.006700583267956972
Test R2   : 0.9961050062530086
Test RMSE : 3834.253925784544 원
걸린시간  : 26.12 초
★ 2026-10-13 삼성전자 시가 예측값 : 265700.53 원 ★
keras701012_1614_0060-0.0002.keras

9차 (최적 🥇)
loss(MSE) : 0.00011265902867307886
MAE       : 0.005798144266009331
Test R2   : 0.9968995005123993
Test RMSE : 3420.923169424128 원
걸린시간  : 65.9 초
★ 2026-10-13 삼성전자 시가 예측값 : 269535.3 원 ★
keras701012_1615_0138-0.0001.keras

10차
loss(MSE) : 0.00013801379827782512
MAE       : 0.0068540386855602264
Test R2   : 0.9962017102316575
Test RMSE : 3786.356790146433 원
걸린시간  : 83.88 초
★ 2026-10-13 삼성전자 시가 예측값 : 270166.6 원 ★
keras701012_1618_0191-0.0002.keras

11차
loss(MSE) : 0.0001714338141027838
MAE       : 0.007683346513658762
Test R2   : 0.9952819562740376
Test RMSE : 4219.961310507074 원
걸린시간  : 43.29 초
★ 2026-10-13 삼성전자 시가 예측값 : 272145.03 원 ★
keras701012_1620_0078-0.0001.keras
'''