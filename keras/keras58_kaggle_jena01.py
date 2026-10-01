'''
y는 wd (deg)로 하고, 자르는건 마음대로 (y : 144개)
맞추기 : 2016.12.31 00시 10분 ~ 2017.01.01 0시까지 데이터 144개는 훈련에 사용 X
'''
# y를 T (degC)의 경우 rmse로 최고 1.48

import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'      # 메모리 모으기
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
import time
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


start_time1 = time.time()                      
path = './_data/kaggle_jena/'
datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
# print(datasets.shape)       #(420551, 14)

y_cor = datasets[-144:]['T (degC)']      # 예측치 정답 데이터
# print(y_cor.shape)      #(144,)

##################### 훈련 데이터 자르기 #####################
x_data = datasets[:-288].drop(['T (degC)'], axis=1)    
y_data = datasets[144:-144]['T (degC)']    


# print(x_data.shape)    #(420263, 13)
# print(y_data.shape)    #(420263,)


def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size +1) :
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

size_x=144
size_y=144

x= split_x(x_data,size_x)
y= split_x(y_data,size_y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333, shuffle=False)


x_train = x_train.reshape(-1, 13)
x_test = x_test.reshape(-1,13)

scaler = RobustScaler()
# scaler = MinMaxScaler() 
# scaler = StandardScaler()
# scaler = MaxAbsScaler()         

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 144, 13)
x_test = x_test.reshape(-1, 144, 13)
end_time1 = time.time()




np_path = './_data/kaggle_jena_npy/'
np.save(np_path + 'y_cor.npy', arr=y_cor)
np.save(np_path + 'x_train.npy', arr=x_train)
np.save(np_path + 'y_train.npy', arr=y_train)
np.save(np_path + 'x_test.npy', arr=x_test)
np.save(np_path + 'y_test.npy', arr=y_test)


print('걸린시간1 :', round(end_time1 - start_time1, 2), '초')

# 걸린시간1 : 75.2 초