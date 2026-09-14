from tensorflow.keras.models import Sequential, Model   #models에서 Model은 그냥 함수형 모델이라고 생각하면 된다.
from tensorflow.keras.layers import Dense,Dropout, Input 

#2-1. 순차적 모델
model = Sequential()
model.add(Dense(10, input_shape=(3,)))
model.add(Dropout(.2))
model.add(Dense(9))
model.add(Dropout(.2))
model.add(Dense(1))

model.summary()
################################################
#2-2. 함수형 모델 ----------> 순차적 모델과는 다르게 모델을 나중에 정의한다.
input1 = Input(shape=(3,))      # 순차적 모델에서 'model.add(Dense(10, input_shape=(3,)))'에 해당 됨.
dense1 = Dense(10, name='ys1')(input1)       # (input1)을 쓰므로 15번 줄과 16번 줄이 연결 되었다.
drop1 = Dropout(.2)(dense1)
dense2 = Dense(9, name='ys2')(drop1)         # Dense의 이름은 굳이 안해도 된다. 다만, model.summary()에서 가독성을 높이기 위함이다.
drop2 = Dropout(.2)(dense2)
output1 = Dense(1)(drop2)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()
