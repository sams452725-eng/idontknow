# keras69-1 카피

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

#1. 데이터
x1 = np.array([range(100), range(301,401)]).T
            # 삼성종가      하이닉스 종가
x2 = np.array([range(101,201), range(411,511), range(150,250)]).transpose()
            #    원유가             환율          금시세
x3 = np.array([range(100), range(301,401), range(77,177), range(33,133)]).transpose()

y = np.array(range(3001, 3101))
            # 화성의 화씨 온도

# print(x1.shape, x2.shape, x3.shape)       # (100, 2) (100, 3) (100, 4)

# exit()
x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516), range(249,255)]).transpose()
x3_pred = np.array([range(100,106), range(400,406), range(177,183), range(133,139)]).transpose()



x1_train, x1_test, y_train, y_test = train_test_split(x1, y, train_size=0.7, random_state=333)
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, train_size=0.7, random_state=333)
x3_train, x3_test, y_train, y_test = train_test_split(x3, y, train_size=0.7, random_state=333)




#2-1. 모델

input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='han1')(input1)
dense2 = Dense(20, activation='relu', name='han2')(dense1)
dense3 = Dense(30, activation='relu', name='han3')(dense2)
output1 = Dense(5, activation='relu', name='han4')(dense3)
# model1 = Model(inputs=input1, outputs=output1)

#2-2. 모델

input21 = Input(shape=(3,))
dense21 = Dense(50, name='han21')(input21)
dense22 = Dense(40, name='han22')(dense21)
dense23 = Dense(30, name='han23')(dense22)
dense24 = Dense(20, name='han24')(dense23)
output21 = Dense(3, name='han25')(dense24)
# model2 = Model(inputs=input21, outputs=output21)

#2-3. 모델

input31 = Input(shape=(4,))
dense31 = Dense(80, name='han31')(input31)
dense32 = Dense(60, name='han32')(dense31)
dense33 = Dense(40, name='han33')(dense32)
dense34 = Dense(20, name='han34')(dense33)
output31 = Dense(4, name='han35')(dense34)


#2-3. 모델 합치기
from tensorflow.keras.layers import concatenate, Concatenate      # 대문자 Concat과 소문자 concat은 서로 약간의 차이가 있지만 결국 동일한 효과다.

# merge1 = concatenate([output1, output21], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output21, output31])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input21, input31], outputs=last_output)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit([x1_train,x2_train,x3_train], y_train, epochs=300, batch_size=8, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)
print("=================================================")
print('loss :', loss)
print("=================================================")

y_pred = model.predict([x1_test, x2_test, x3_test])          

y_pred_result = model.predict([x1_pred, x2_pred, x3_pred])


print("예측 확률값 :", y_pred_result)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
print("r2: ", r2)

mse = mean_squared_error(y_test, y_pred)
print("mse: ", mse)

def RMSE(y_test, y_predict):  #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)
print("RMSE : ", rmse)




'''
1차
loss : 0.2741168737411499
예측 확률값 : [3091.7788]
 [3092.7917]
 [3093.805 ]
 [3094.8179]
 [3095.8313]]
2차
loss : 1.5894572769070692e-08
예측 확률값 : [[3092.7505]
 [3093.7505]
 [3094.7505]
 [3095.7505]
 [3096.7507]
 [3097.7505]]
3차
loss : 0.5766924023628235
예측 확률값 : [[3091.6677]
 [3092.6729]
 [3093.6782]
 [3094.6838]
 [3095.6887]
 [3096.6938]]
r2:  0.9990228414535522
mse:  0.5766924619674683
RMSE :  0.759402700790212







'''





