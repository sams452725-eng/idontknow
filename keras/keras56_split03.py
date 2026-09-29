import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import time
from sklearn.model_selection import train_test_split


a = np.array(range(1,101))

x_predict = np.array(range(96,106))   # 101~106까지 찾자

size = 6

# loss 지표는 0.1 이하
# 결과는
# [101,102,103,104,105,106]의 근사치가 나오면 됨


def split_x(dataset, size):           
    aaa = []
    for i in range(len(dataset) - size +1) :
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)

x = bbb[:, :-1]
y = bbb[:, -1]
x = x.reshape(x.shape[0], x.shape[1], 1)

# print(x,y, x.shape,y.shape)    # (95, 5, 1) (95,)

# exit()


#2. 모델구성
model = Sequential()
# model.add(GRU(64,input_shape=(5, 1)))            
# model.add(LSTM(64, input_shape=(5, 1)))        
model.add(SimpleRNN(64, input_shape=(5, 1), activation='linear'))   
model.add(Dense(16, activation='linear'))
model.add(Dense(8, activation='linear'))
model.add(Dense(1))

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

#4. 평가,예측
print('================= model.evaluate =================')
loss = model.evaluate(x, y)
print('loss :', loss) 
print('걸린시간 :', round(end_time - start_time, 2), '초')

predict_split = split_x(x_predict, 5)    #(6, 5)      
predict_split = predict_split.reshape(predict_split.shape[0], predict_split.shape[1], 1) 
result = model.predict(predict_split)
print('================= 101~106 예측 결과 =================')
print(result)
'''
1차
[[81.27947]
 [81.41776]
 [81.52693]
 [81.65623]
 [81.77153]
 [81.86988]]
loss : 158.0982208251953
걸린시간 : 82.45 초

 2차
[[38.976585]
 [38.949017]
 [38.924892]
 [38.89792 ]
 [38.87137 ]
 [38.84253 ]]
 loss : 2708.22412109375
걸린시간 : 4.35 초

 3차
[[68.865295]
 [68.76326 ]
 [68.67179 ]
 [68.569756]
 [68.44184 ]
 [68.33097 ]]
loss : 496.8953552246094
걸린시간 : 95.71 초

4차
[[72.972435]
 [72.988686]
 [72.99168 ]
 [73.02139 ]
 [73.02139 ]
 [73.03651 ]]
loss : 366.7445983886719
걸린시간 : 17.96 초

5차
[[76.89622 ]
 [76.948105]
 [76.99845 ]
 [77.02887 ]
 [77.07588 ]
 [77.110756]]
loss : 249.7172088623047
걸린시간 : 44.93 초

6차
[[75.48783]
 [75.535  ]
 [75.57394]
 [75.61066]
 [75.64346]
 [75.66795]]
loss : 288.666015625
걸린시간 : 137.1 초

7차
[[69.13452 ]
 [69.20684 ]
 [69.26684 ]
 [69.34649 ]
 [69.423904]
 [69.48321 ]]
loss : 551.3480834960938
걸린시간 : 7.18 초

8차
[[79.6179  ]
 [79.844055]
 [80.07303 ]
 [80.31049 ]
 [80.50528 ]
 [80.70713 ]]
loss : 50.230125427246094
걸린시간 : 6.82 초

9차
[[ 99.72186 ]
 [100.432686]
 [101.076645]
 [101.67922 ]
 [102.216705]
 [102.71443 ]]
loss : 0.048014696687459946
걸린시간 : 32.98 초

10차
[[100.725044]
 [101.4662  ]
 [102.143364]
 [102.747734]
 [103.27206 ]
 [103.73515 ]]
loss : 0.004621441010385752
걸린시간 : 15.35 초

11차
[[100.42569 ]
 [100.94086 ]
 [101.371315]
 [101.734726]
 [102.03352 ]
 [102.2898  ]]
 loss : 0.002323317574337125
걸린시간 : 11.83 초

12차
[[100.63675 ]
 [101.273674]
 [101.78721 ]
 [102.18932 ]
 [102.50431 ]
 [102.74367 ]]
loss : 0.0009616765310056508
걸린시간 : 13.04 초

















































'''