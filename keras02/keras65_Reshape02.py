# 36-2 카피

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist     #CNN을 하는 사람들은 항상 mnist를 쓴다.
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, GlobalAveragePooling2D, Input
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255.
x_test = x_test/255. 


from tensorflow.keras.layers import Reshape

#2. 모델구성
model = Sequential()
# model.add(Dense(280, input_shape=(28,28)))
# model.add(Reshape(target_shape=(28,28,10)))
# model.add(Conv2D(32, (3,3), activation='relu'))
# model.add(Conv2D(16, (3,3), activation='relu'))
# model.add(GlobalAveragePooling2D()) 
# model.add(Dense(units=32, activation='relu'))
# model.add(Dense(10, activation='softmax'))

input = Input(shape=(28,28))

dense = Dense(280)(input)
reshape = Reshape(target_shape=(28,28,10))(dense)
conv1 = Conv2D(32,(3,3), activation='relu')(reshape)
conv2 = Conv2D(32,(3,3), activation='relu')(conv1)
gap = GlobalAveragePooling2D()(conv2)
dense2 = Dense(32, activation='relu')(gap)
output = Dense(10, activation='softmax')(dense2)

model = Model(inputs=input, outputs=output)


# model.summary()

# exit()
#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['acc'])

start_time=time.time()
model.fit(x_train, y_train, epochs=50, batch_size=128,
          verbose=1,
          validation_split=.2,
          )
end_time = time.time()

#4. 평가,예측
print('================= model.evaluate =================')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1], 4))



# exit()
y_predict = model.predict(x_test)


y_predict = np.argmax(y_predict, axis=1)

# print(y_test.shape, y_predict.shape)
# exit()
acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time - start_time, 2), '초')


# 0.995 맞추기
'''
loss : 0.06397169083356857
acc : 0.9807
accuracy_score :  0.9807
걸린시간 :  269.89 초




'''






