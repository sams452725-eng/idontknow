import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
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
# model.add(SimpleRNN(64, input_shape=(3, 1), return_sequences=True, activation='relu'))        
model.add(GRU(64,input_shape=(3,1), return_sequences=True, activation='relu'))            
# model.add(LSTM(64, input_shape=(3,1), return_sequences=True, activation='linear'))    #ValueError: Input 0 of layer "lstm_1" is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (None, 10) 때문에 return을 해준다.
# model.add(LSTM(32, activation='linear', return_sequences=True))      # 원래라면  Output Shape는 (, 10)의 벡터 형태로 나오는데 LSTM의 input_shape는 2차원 매트릭스 형태를 넣어줘야 하기 때문에 return을 써서 hidden의 모든 Output Shape를 다 꺼낸다.
model.add(LSTM(16, activation='linear', return_sequences=True))      
model.add(LSTM(8, activation='linear'))      
model.add(Dense(8, activation='relu'))
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
        batch_size = 4,
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
1차
loss : 0.00042329778079874814
[50,60,70]의 결과 : [[84.47601]]
2차
loss : 0.0005459457170218229
[50,60,70]의 결과 : [[83.82252]]
3차
loss : 0.00033116049598902464
[50,60,70]의 결과 : [[81.780624]]
4차
loss : 0.00034508982207626104
[50,60,70]의 결과 : [[80.46818]]
5차
loss : 0.0005533251096494496
[50,60,70]의 결과 : [[80.821144]]


지금은 선생님이 80에 맞추라는게 요구였으니까 80에 더 가까운 결과를 선택해야되지만,
개인적으로 판단을 해야될때는 당연하게도 loss 값이 더 낮은 결과를 선택해야한다.



'''