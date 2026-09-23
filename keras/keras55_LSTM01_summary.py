# 54-2 카피
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],       
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ])
y = np.array([4,5,6,7,8,9,10])


x = x.reshape(x.shape[0], x.shape[1], 1)   # (7, 3, 1)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(10, input_shape=(3, 1)))       #SimpleRNN의 단점은 데이터의 개수가 많으면 오랜 과거의 W(가중치)가 다음 예측에 반영이 안된다. 그걸 조금이라도 살리기 위해서 LSTM이 나왔다.
model.add(LSTM(10, input_shape=(3, 1)))       #LSTM은 RNN 계열에서 가장 많이 쓰인다. 하지만 LSTM은 연산량이 많고 속도가 느리다. 그래서 'LSTM은 죽었다' 하고 Transformer가 등장했다.
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()
# 파라미터의 개수 = units*feature + units*bias + units*units


'''
SimpleRNN
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 10)                120       
                                                                 
 dense (Dense)               (None, 7)                 77        
                                                                 
 dense_1 (Dense)             (None, 1)                 8         
                                                                 
=================================================================
Total params: 205
Trainable params: 205
Non-trainable params: 0
_________________________________________________________________


LSTM
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 10)                480       
                                                                 
 dense (Dense)               (None, 7)                 77        
                                                                 
 dense_1 (Dense)             (None, 1)                 8         
                                                                 
=================================================================
Total params: 565
Trainable params: 565
Non-trainable params: 0
_________________________________________________________________




'''




























































































