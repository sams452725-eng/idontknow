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
# print(x1, x2)        #(100, 2) (100, 3)

y = np.array(range(3001, 3101))
            # 화성의 화씨 온도

x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516), range(249,255)]).transpose()



x1_train, x1_test, y_train, y_test = train_test_split(x1, y, train_size=0.7, random_state=333)
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, train_size=0.7, random_state=333)




#2-1. 모델

input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='han1')(input1)
dense2 = Dense(20, activation='relu', name='han2')(dense1)
dense3 = Dense(30, activation='relu', name='han3')(dense2)
output1 = Dense(5, activation='relu', name='han4')(dense3)
# model1 = Model(inputs=input1, outputs=output1)

#2-1. 모델

input21 = Input(shape=(3,))
dense21 = Dense(50, name='han21')(input21)
dense22 = Dense(40, name='han22')(dense21)
dense23 = Dense(30, name='han23')(dense22)
dense24 = Dense(20, name='han24')(dense23)
output21 = Dense(3, name='han25')(dense24)
# model2 = Model(inputs=input21, outputs=output21)


#2-3. 모델 합치기
from tensorflow.keras.layers import concatenate, Concatenate      # 대문자 Concat과 소문자 concat은 서로 약간의 차이가 있지만 결국 동일한 효과다.

# merge1 = concatenate([output1, output21], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output21])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input21], outputs=last_output)
# model.summary()

# Model: "functional"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │ input_layer_1 (InputLayer)    │ (None, 3)                 │               0 │ -                          │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ input_layer (InputLayer)      │ (None, 2)                 │               0 │ -                          │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han21 (Dense)                 │ (None, 50)                │             200 │ input_layer_1[0][0]        │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han1 (Dense)                  │ (None, 10)                │              30 │ input_layer[0][0]          │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han22 (Dense)                 │ (None, 40)                │           2,040 │ han21[0][0]                │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han2 (Dense)                  │ (None, 20)                │             220 │ han1[0][0]                 │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han23 (Dense)                 │ (None, 30)                │           1,230 │ han22[0][0]                │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han3 (Dense)                  │ (None, 30)                │             630 │ han2[0][0]                 │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han24 (Dense)                 │ (None, 20)                │             620 │ han23[0][0]                │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han4 (Dense)                  │ (None, 5)                 │             155 │ han3[0][0]                 │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han25 (Dense)                 │ (None, 3)                 │              63 │ han24[0][0]                │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ mg1 (Concatenate)             │ (None, 8)                 │               0 │ han4[0][0], han25[0][0]    │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ mg2 (Dense)                   │ (None, 10)                │              90 │ mg1[0][0]                  │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ mg3 (Dense)                   │ (None, 5)                 │              55 │ mg2[0][0]                  │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ last (Dense)                  │ (None, 1)                 │               6 │ mg3[0][0]                  │
# └───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
#  Total params: 5,339 (20.86 KB)
#  Trainable params: 5,339 (20.86 KB)
#  Non-trainable params: 0 (0.00 B)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit([x1_train,x2_train], y_train, epochs=100, batch_size=8, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print("=================================================")
print('loss :', loss)
print('acc :', round(loss, 4))
print("=================================================")

y_pred = model.predict([x1_test, x2_test])          
y_pred = np.round(y_pred)

y_pred_result = model.predict([x1_pred, x2_pred])

acc_score = accuracy_score(y_test, y_pred)

print('acc_score :', acc_score)
print("예측 확률값 :", y_pred_result)


'''
loss : 0.38264864683151245
acc : 0.3826
acc_score : 0.5
예측 확률값 : [[3101.3367]
 [3102.3594]
 [3103.3816]
 [3104.4045]
 [3105.4263]
 [3106.4495]]

'''





