import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'      # 메모리 모으기
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
import time
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score

#1. 데이터

start_time1 = time.time()                      
np_path = './_data/kaggle_jena_npy/'

y_cor = np.load(np_path + 'y_cor.npy')
x_train = np.load(np_path + 'x_train.npy')
y_train = np.load(np_path + 'y_train.npy')
x_test = np.load(np_path + 'x_test.npy')
y_test = np.load(np_path + 'y_test.npy')

end_time1 = time.time()

# print('걸린시간1 :', round(end_time1 - start_time1, 2), '초')    #걸린시간1 : 2.1 초


#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(64, input_shape=(3, 1), return_sequences=True, activation='linear'))        
model.add(GRU(64,input_shape=(144,13), return_sequences=True))            
# model.add(LSTM(32, input_shape=(3,1), return_sequences=True, activation='linear'))    
model.add(LSTM(32, return_sequences=True))      
# model.add(LSTM(32, input_shape=(144,13), activation='relu', return_sequences=True))
# model.add(Flatten())     
model.add(LSTM(8,))      
model.add(Dense(4,))
model.add(Dense(144))
# model.summary()


#3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam
# learning_rate = 0.005
learning_rate = 0.05

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate), metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
        monitor='loss',
        mode='auto',
        patience=50,
        restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
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
    mode='auto',
    save_best_only=True,
    filepath = filepath,             
    verbose=1,
)
start_time2 = time.time()                      
model.fit(x_train, y_train, 
        epochs = 1500,
        batch_size = 1024,
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












'''