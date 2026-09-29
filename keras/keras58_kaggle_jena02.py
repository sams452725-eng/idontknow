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

#1. 데이터
np_path = './_data/kaggle_jena_npy/'
y_cor = np.load(np_path + 'keras58_y_cor.npy')
y_data = np.load(np_path + 'keras58_y_data.npy')
x_data = np.load(np_path + 'keras58_x_data.npy')

# print(x_data.shape, y_data.shape)    #(420263, 13) (420263,)


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

# print(x.shape, y.shape)   #(420120, 144, 13) (420120, 144)

# exit()
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333,)

# print(x_test.shape, x_train.shape)   #(84024, 144, 13) (336096, 144, 13)
# print(y_test.shape, y_train.shape)   #(84024, 144) (336096, 144)
# exit()

scaler = RobustScaler()
# scaler = MinMaxScaler() 
# scaler = StandardScaler()
# scaler = MaxAbsScaler()                                      

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)




exit()
#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(64, input_shape=(3, 1), return_sequences=True, activation='linear'))        
# model.add(GRU(64,input_shape=(3,1), return_sequences=True, activation='linear'))            
# model.add(LSTM(32, input_shape=(3,1), return_sequences=True, activation='linear'))    
# model.add(LSTM(32, activation='linear', return_sequences=True))      
model.add(LSTM(32, input_shape=(144,13), activation='linear', return_sequences=True)), #return_sequences=True)) 
# model.add(Flatten())     
model.add(LSTM(8, activation='linear'))      
model.add(Dense(4, activation='linear'))
model.add(Dense(1, activation='relu'))
# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
        monitor='loss',
        mode='min',
        patience=50,
        restore_best_weights=True,
)
start_time = time.time()                      
model.fit(x_train, y_train, 
        epochs = 1500,
        batch_size = 1024,
        callbacks=[es],
        validation_split=0.3,

)
end_time = time.time()


#4. 평가, 예측n
results = model.evaluate(x_test,y_test)
print('loss :', results)

y_predict = model.predict(x_cor)


print('y_predict :', y_predict)