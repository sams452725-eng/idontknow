import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'      # 메모리 모으기
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

start_time1 = time.time()                      
np_path = './_save/kaggle_jena_npy/'

y_cor = np.load(np_path + 'y_cor.npy')
x_train = np.load(np_path + 'x_train.npy')
y_train = np.load(np_path + 'y_train.npy')
x_test = np.load(np_path + 'x_test.npy')
y_test = np.load(np_path + 'y_test.npy')

end_time1 = time.time()

# print('걸린시간1 :', round(end_time1 - start_time1, 2), '초')    #걸린시간1 : 2.1 초

# print(x_train.shape, x_test.shape)   #(336096, 144, 13) (84024, 144, 13)

# x_train = x_train.reshape(-1, 12, 12, 13)
# x_test = x_test.reshape(-1, 12, 12, 13)

# print(x_train.shape, x_test.shape)   #(336096, 12, 12, 13) (84024, 12, 12, 13)

# exit()
#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, input_shape=(144,13), activation='relu', padding='same'))
model.add(Conv1D(32,3, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(Dropout(.2))
# model.add(Flatten())
# model.add(GlobalAveragePooling1D())
model.add(LSTM(32, return_sequences=True))      
model.add(LSTM(16))      
model.add(Dense(32, activation='relu'))
model.add(Dense(144,))
# model.summary()


#3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam
# learning_rate = 0.005
learning_rate = 0.001

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
    patience=7,
    verbose=1,
    factor=0.5,        
)  
import datetime
date = datetime.datetime.now()    

date = date.strftime('%m%d_%H%M')

path = './_save/kaggle_jena/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'     
filepath = ''.join([path, 'jena_', date,'_', filename])


mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='min',
    save_best_only=True,
    filepath = filepath,             
    verbose=1,
)
start_time2 = time.time()                      
model.fit(x_train, y_train, 
        epochs = 100,
        batch_size = 2048,
        callbacks=[es,mcp,rlr],
        validation_split=0.3,

)
end_time2 = time.time()

# exit()
#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss(MSE) :', loss[0])
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


print('걸린시간1 :', round(end_time1 - start_time1, 2), '초')
print('걸린시간2 :', round(end_time2 - start_time2, 2), '초')



'''
1차
loss(MSE) : 10.127513885498047
MAE       : 2.452821969985962
mse:  10.127504760522212
Test R2   : 0.8445206541705341
Test RMSE : 3.182374076145388
걸린시간1 : 4.6 초
걸린시간2 : 4427.52 초

CNN----->LSTM
1차
loss(MSE) : 21.872039794921875
MAE       : 3.776445150375366
mse:  21.87205188216139
Test R2   : 0.6641342864356186
Test RMSE : 4.676756555793918
걸린시간1 : 1.88 초
걸린시간2 : 288.87 초

2차
loss(MSE) : 9.15666389465332
MAE       : 2.3327317237854004
mse:  9.156667056894243
Test R2   : 0.8594281940035993
Test RMSE : 3.0259985222888397
걸린시간1 : 2.86 초
걸린시간2 : 152.02 초

Conv1D
1차
loss(MSE) : 3584.376953125
MAE       : 5.650672912597656
mse:  3584.3772356781606
Test R2   : -54.048097504951016
Test RMSE : 59.86966874535185
걸린시간1 : 1.88 초
걸린시간2 : 756.39 초

2차
loss(MSE) : 332.6961669921875
MAE       : 2.7016682624816895
mse:  332.69646382732964
Test R2   : -4.111387736398615
Test RMSE : 18.239968854889245
걸린시간1 : 2.9 초
걸린시간2 : 1471.19 초

3차
loss(MSE) : 1695.49658203125
MAE       : 3.5272927284240723
mse:  1695.496170866028
Test R2   : -25.048740669389865
Test RMSE : 41.17640308314979
걸린시간1 : 2.68 초
걸린시간2 : 114.6 초

4차
loss(MSE) : 7.763662815093994
MAE       : 2.141594886779785
mse:  7.763669312198924
Test R2   : 0.8808210357400471
Test RMSE : 2.7863361807576132
걸린시간1 : 2.08 초
걸린시간2 : 139.35 초











'''