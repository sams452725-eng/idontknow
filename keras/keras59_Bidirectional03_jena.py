import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'      # 메모리 모으기
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
import time
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, Flatten, Bidirectional
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
# model.add(Bidirectional(SimpleRNN(64), input_shape=(144, 13), return_sequences=True, activation='relu'))        
model.add(Bidirectional(GRU(64, return_sequences=True, activation='relu'),input_shape=(144,13)))            
# model.add(Bidirectional(LSTM(64), input_shape=(3,1), return_sequences=True, activation='linear'))    
model.add(Bidirectional(LSTM(32), activation='relu', return_sequences=True))      
# model.add(LSTM(32, input_shape=(144,13), activation='relu', return_sequences=True))
# model.add(Flatten())     
model.add(LSTM(8, activation='relu'))      
model.add(Dense(4, activation='relu'))
model.add(Dense(144, activation='relu'))
# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
        monitor='loss',
        mode='auto',
        patience=50,
        restore_best_weights=True,
)
start_time2 = time.time()                      
model.fit(x_train, y_train, 
        epochs = 1500,
        batch_size = 1024,
        callbacks=[es],
        validation_split=0.3,

)
end_time2 = time.time()

exit()
#4. 평가, 예측n
loss = model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)

def RMSE(y_test, y_predict):  #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("rmse : ", rmse)


print('걸린시간1 :', round(end_time1 - start_time1, 2), '초')
print('걸린시간2 :', round(end_time2 - start_time2, 2), '초')




'''
1차















'''