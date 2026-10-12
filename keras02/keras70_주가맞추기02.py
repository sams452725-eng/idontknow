import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
import time
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
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


#2. 모델


start_time2 = time.time()                      
path = './_save/keras70/'

model = load_model(path +'keras701012_1615_0138-0.0001.keras')

end_time2 = time.time()



#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)
print('loss(MSE) :', loss[0])
print('MAE       :', loss[1])

y_predict = model.predict([x1_test, x2_test, x3_test])

y_test_before_scale = scaler.inverse_transform(y_test)
y_predict_before_scale = scaler.inverse_transform(y_predict)
mse = mean_squared_error(y_test_before_scale, y_predict_before_scale)

def RMSE(y_test, y_predict):
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


