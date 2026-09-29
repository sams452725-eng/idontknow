import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import time

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70]) 

# print(x.shape, y.shape)       # (13, 3) (13,)

x = x.reshape(x.shape[0], x.shape[1], 1)   # (13, 3, 1)

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(64, input_shape=(3, 1), return_sequences=True, activation='linear'))        
model.add(GRU(64,input_shape=(3,1), return_sequences=True, activation='linear'))            
model.add(LSTM(32, input_shape=(3,1), return_sequences=True, activation='linear'))    #ValueError: Input 0 of layer "lstm_1" is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (None, 10) 때문에 return을 해준다.
# model.add(LSTM(32, activation='linear', return_sequences=True))      # 원래라면  Output Shape는 (, 10)의 벡터 형태로 나오는데 LSTM의 input_shape는 2차원 매트릭스 형태를 넣어줘야 하기 때문에 return을 써서 hidden의 모든 Output Shape를 다 꺼낸다.
model.add(LSTM(32, activation='linear', return_sequences=True)) 
# model.add(Flatten())     
model.add(LSTM(8, activation='linear'))      
model.add(Dense(8, activation='linear'))
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
model.fit(x, y, 
        epochs = 1500,
        batch_size = 3,
        callbacks=[es],
)
end_time = time.time()


#4. 평가, 예측n
results = model.evaluate(x,y)
print('loss :', results)

x_predict = np.array([50,60,70]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[50,60,70]의 결과 :', y_predict)

'''
keras57-01의 best
loss : 0.00034508982207626104
[50,60,70]의 결과 : [[80.46818]]

1차
loss : 0.0013617274817079306
[50,60,70]의 결과 : [[80.94963]]
2차
loss : 0.0021307754795998335
[50,60,70]의 결과 : [[81.22052]]
3차
loss : 0.004643223248422146
[50,60,70]의 결과 : [[83.07686]]
4차
loss : 0.030098892748355865
[50,60,70]의 결과 : [[80.28504]]
5차
loss : 0.0031210817396640778
[50,60,70]의 결과 : [[80.80236]]
6차
loss : 0.0005094382213428617
[50,60,70]의 결과 : [[81.797005]]
7차
loss : 0.0012300872476771474
[50,60,70]의 결과 : [[82.2877]]
8차
loss : 0.006659334525465965
[50,60,70]의 결과 : [[81.02575]]
9차
loss : 0.00398680567741394
[50,60,70]의 결과 : [[81.51499]]



SimpleRNN,GRU,LSTM
3개의 layer는 모두 return_sequence를 통해서 엮어서 사용할수있다.
하지만 선생님 왈 : 그런건 본적도 시도 해본적도 없다.

'''