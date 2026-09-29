'''
y는 wd (deg)로 하고, 자르는건 마음대로 (y : 144개)
맞추기 : 2016.12.31 00시 10분 ~ 2017.01.01 0시까지 데이터 144개는 훈련에 사용 X
'''

import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'      # 메모리 모으기
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

path = './_data/kaggle_jena/'
datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
# print(datasets.shape)       #(420551, 14)

y_cor = datasets[-144:]['wd (deg)']      # 예측치 정답 데이터
# print(y_cor.shape)      #(144,)

##################### 훈련 데이터 자르기 #####################
x_data = datasets[:-288].drop(['wd (deg)'], axis=1)    
y_data = datasets[144:-144]['wd (deg)']    


# print(x_data.shape)    #(420263, 13)
# print(y_data.shape)    #(420263,)

np_path = './_data/kaggle_jena_npy/'
np.save(np_path + 'keras58_y_cor.npy', arr=y_cor)
np.save(np_path + 'keras58_x_data.npy', arr=x_data)
np.save(np_path + 'keras58_y_data.npy', arr=y_data)

