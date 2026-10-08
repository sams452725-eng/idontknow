# keras69-2 카피

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

y1 = np.array(range(3001, 3101))
            # 화성의 화씨 온도
y2 = np.array(range(13001, 13101))

# print(x1.shape, x2.shape, x3.shape)       # (100, 2) (100, 3) (100, 4)
# print(y1.shape, y2.shape)       # (100,) (100,)


# exit()
x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516), range(249,255)]).transpose()
x3_pred = np.array([range(100,106), range(400,406), range(177,183), range(133,139)]).transpose()



x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.7, random_state=333)
x2_train, x2_test, y1_train, y1_test = train_test_split(x2, y1, train_size=0.7, random_state=333)
x3_train, x3_test, y1_train, y1_test = train_test_split(x3, y1, train_size=0.7, random_state=333)

x1_train, x1_test, y2_train, y2_test = train_test_split(x1, y2, train_size=0.7, random_state=333)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.7, random_state=333)
x3_train, x3_test, y2_train, y2_test = train_test_split(x3, y2, train_size=0.7, random_state=333)




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

#2-4. 모델 합치기
from tensorflow.keras.layers import concatenate, Concatenate      # 대문자 Concat과 소문자 concat은 서로 약간의 차이가 있지만 결국 동일한 효과다.

# merge1 = concatenate([output1, output21], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output21, output31])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)

# 2-5. 분기1
last_dense1 = Dense(10, name='ld1')(merge3)
last_dense2 = Dense(10, name='ld2')(last_dense1)
last_output1 = Dense(1, name='last1')(last_dense2)

# 2-6 분기2
last_dense21 = Dense(10, name='ld21')(merge3)
last_dense22 = Dense(10, name='ld22')(last_dense21)
last_output2 = Dense(1, name='last2')(last_dense22)

model = Model(inputs=[input1, input21, input31], outputs=[last_output1, last_output2])




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit([x1_train,x2_train,x3_train], [y1_train, y2_train], epochs=300, batch_size=8, validation_split=0.2)

#4. 평가, 예측
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print("=================================================")
print('loss :', loss)
print("=================================================")

y_pred = model.predict([x1_test, x2_test, x3_test])          

y_pred_result = model.predict([x1_pred, x2_pred, x3_pred])


print("예측 확률값 :", y_pred_result)

from sklearn.metrics import r2_score, mean_squared_error
# 1. 첫 번째 출력(y1) 채점
r2_1 = r2_score(y1_test, y_pred[0])
mse_1 = mean_squared_error(y1_test, y_pred[0])
rmse_1 = np.sqrt(mse_1)
# 2. 두 번째 출력(y2) 채점
r2_2 = r2_score(y2_test, y_pred[1])
mse_2 = mean_squared_error(y2_test, y_pred[1])
rmse_2 = np.sqrt(mse_2)
# 3. 각각 출력 및 평균 점수 확인
print("================= R2 SCORE =================")
print("y1 R2 :", r2_1)
print("y2 R2 :", r2_2)
print("평균 R2 :", (r2_1 + r2_2) / 2)
print("================= RMSE =================")
print("y1 RMSE :", rmse_1)
print("y2 RMSE :", rmse_2)
print("평균 RMSE :", (rmse_1 + rmse_2) / 2)


'''
1차
loss : [0.13024163246154785, 0.12175574153661728, 0.008485889062285423]
예측 확률값 : [array([[3092.1199],
       [3093.107 ],
       [3094.0928],
       [3095.0781],
       [3096.0637],
       [3097.0503]], dtype=float32), array([[13064.486],
       [13065.488],
       [13066.493],
       [13067.496],
       [13068.498],
       [13069.501]], dtype=float32)]
2차
loss : [0.5072862505912781, 0.0702550932765007, 0.43703117966651917]
예측 확률값 : [array([[3092.196 ],
       [3093.2375],
       [3094.2803],
       [3095.302 ],
       [3096.3044],
       [3097.3076]], dtype=float32), array([[13061.58 ],
       [13063.094],
       [13064.607],
       [13066.149],
       [13067.717],
       [13069.289]], dtype=float32)]
3차
loss : [0.6373454928398132, 0.6029719710350037, 0.034373536705970764]
예측 확률값 : [array([[3090.0457],
       [3091.015 ],
       [3091.9846],
       [3092.9536],
       [3093.9243],
       [3094.8933]], dtype=float32), array([[13058.627],
       [13059.633],
       [13060.639],
       [13061.646],
       [13062.654],
       [13063.66 ]], dtype=float32)]
4차
loss : [0.0012450178619474173, 0.0011749863624572754, 7.003148493822664e-05]
예측 확률값 : [array([[3090.2744],
       [3091.273 ],
       [3092.272 ],
       [3093.2705],
       [3094.269 ],
       [3095.268 ]], dtype=float32), array([[13054.102 ],
       [13055.104 ],
       [13056.103 ],
       [13057.1045],
       [13058.103 ],
       [13059.1045]], dtype=float32)]
================= R2 SCORE =================
y1 R2 : 0.9999980330467224
y2 R2 : 0.9999998807907104
평균 R2 : 0.9999989569187164
================= RMSE =================
y1 RMSE : 0.034278074077422664
y2 RMSE : 0.008368481638757813
평균 RMSE : 0.02132327785809024



















'''





