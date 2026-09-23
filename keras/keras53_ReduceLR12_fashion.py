# 43-2 카피
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist     
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Input
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist


#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
###################### 여기부터 증폭 ######################
datagen = ImageDataGenerator(
    rescale=1./255,         # 1.은 부동소수점 형변환 1/255. 해도 된다. 어디든 .을 붙이면 된다.--------------->keras43에서 20,21행의 rescale과 같은거다.
    horizontal_flip=True,   # 수평(좌우) 뒤집기
    # vertical_flip=True,     # 수직(상하) 뒤집기    true는 한다, false는 안한다.
    width_shift_range=0.1,  # 평형이동
    # height_shift_range=0.1, 
    rotation_range=15,       # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=0.5,           
    # shear_range=0.7,        # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동(한마디로 찌부)
    fill_mode='nearest'     # 데이터가 이동 및 증폭하면 데이터가 이동한 쪽은 소실되고 그전에 있던 곳은 비워지게 된다. 비워진 곳은 새롭게 채워야 하는데 그때 근처에 있는 수치로 채운다.
)

augment_size=40000

# randidx = np.random.randint(60000, size=augment_size)     #6만개중에 4만개를 랜덤뽑기
# 위의 방법 보다는 아래 방법이 더 좋다
# print(x_train.shape[0])        # 60000
# randidx = np.random.randint(x_train.shape[0], size=augment_size)      #중복뽑기 가능
randidx = np.random.choice(x_train.shape[0], size=augment_size, replace=False)       #중복뽑기 안됨
# print(randidx)
# print(randidx.shape)      # 벡터니까 shape사용 가능
# print(len(randidx))       # list 타입은 len으로 확인해야되는데, 벡터도 가능하다.

# print(np.min(randidx), np.max(randidx))      # 0 59998   -----> max가 59999가 안나온 이유는 랜덤으로 뽑기 때문이다.

x_augmented = x_train[randidx].copy()        # 나중에 python에서 x_augmented를 복사해서 다른 곳에 쓸때 동조화 현상으로 다른 곳에서 바꾼 값이 영향을 미치는걸 미연에 방지
y_augmented = y_train[randidx].copy()

# print(x_augmented.shape, y_augmented.shape)  # (40000, 28, 28) (40000,)
# 40번을 보면 단순 복제된 상태다. 증폭할땐 원래 데이터와 뭐가 달라도 하나라도 달라야 한다. 그렇지 않으면 과적합에 빠진다.

x_augmented = x_augmented.reshape(                               #(40000, 28,28,1)
        x_augmented.shape[0],
        x_augmented.shape[1],
        x_augmented.shape[2], 1)                              
# print(x_augmented.shape)       #(40000, 28, 28, 1)

x_augmented = datagen.flow(
                x_augmented, y_augmented,
                batch_size=augment_size,
                shuffle=False,
).next()[0]

###### 변환완료 ######
# print(x_augmented.shape)       #(40000, 28, 28, 1)

# print(x_train.shape)        #(60000, 28, 28)
x_train = x_train.reshape(-1, 28, 28, 1)   # 또는 (60000, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)     # 또는 (10000, 28, 28, 1)





x_train = np.concatenate((x_train, x_augmented))          #concatenate 정말 많이 쓴다./사슬처럼 엮다라는 의미로 다 차원의 데이터를 엮어준다.
y_train = np.concatenate((y_train, y_augmented))
# print(x_train.shape, y_train.shape)           #(100000, 28, 28, 1) (100000,)
# print(x_test.shape, y_test.shape)             #(10000, 28, 28, 1) (10000,)

# print(np.unique(y_train, return_counts=True))

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
# y_train = y_train.reshape(60000, 1)
y_train = y_train.reshape(-1, 1) 
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)


#2. 모델구성
model = Sequential() 
input1 = Input(shape=(28, 28, 1))

conv1_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input1)
conv1_2 = Conv2D(64, (3, 3), activation='relu')(conv1_1)
pool1   = MaxPooling2D()(conv1_2)
drop1   = Dropout(0.2)(pool1)

conv2_1 = Conv2D(128, (3, 3), padding='same', activation='relu')(drop1)
conv2_2 = Conv2D(128, (3, 3), activation='relu')(conv2_1)
pool2   = MaxPooling2D()(conv2_2)
drop2   = Dropout(0.25)(pool2)

conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu')(drop2)
conv3_2 = Conv2D(256, (3, 3), activation='relu')(conv3_1)
pool3   = MaxPooling2D()(conv3_2)
drop3   = Dropout(0.3)(pool3)

gap    = GlobalAveragePooling2D()(drop3)
dense1  = Dense(512, activation='relu')(gap)
drop4   = Dropout(0.3)(dense1)
dense2  = Dense(256, activation='relu')(drop4)
drop5   = Dropout(0.2)(dense2)

output1 = Dense(10, activation='softmax')(drop5)
model   = Model(inputs=input1, outputs=output1)
# model.summary()


# exit()
#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
# learning_rate = 0.005
# learning_rate = 0.009
learning_rate = 0.01

from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping


model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate),
              metrics=['acc'])
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=13,
        restore_best_weights=True,
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5,        
)     
start_time=time.time()
hist = model.fit(x_train, y_train, epochs=150, batch_size=750,
          verbose=1,
          validation_split=.2,
          callbacks=[es,]
          )
end_time = time.time()


#4. 평가,예측
print('================= model.evaluate =================')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)
y_test = np.argmax(y_test, axis=1).reshape(-1, 1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time - start_time, 2), '초')




# acc 기준 .93 이상
'''
keras43 best
loss : 0.4317833483219147
acc : 0.9269999861717224
accuracy_score :  0.927
걸린시간 :  721.59 초

1차
loss : 0.23419560492038727
acc : 0.9275000095367432
accuracy_score :  0.9275
걸린시간 :  470.84 초

2차
loss : 0.26786214113235474
acc : 0.9194999933242798
accuracy_score :  0.9195
걸린시간 :  413.61 초

3차
loss : 0.3314889967441559
acc : 0.911899983882904
accuracy_score :  0.9119
걸린시간 :  89.09 초

4차
loss : 0.3234211206436157
acc : 0.904699981212616
accuracy_score :  0.9047
걸린시간 :  178.08 초

5차
loss : 0.3301810324192047
acc : 0.9067000150680542
accuracy_score :  0.9067
걸린시간 :  149.71 초

6차
loss : 0.20644204318523407
acc : 0.9301000237464905
accuracy_score :  0.9301
걸린시간 :  1134.69 초





optimizer = learning_rate = 0.005
loss : 0.33358752727508545
acc : 0.8755999803543091
accuracy_score :  0.8756
걸린시간 :  2060.85 초

optimizer = learning_rate = 0.009
loss : 0.38446199893951416
acc : 0.8514000177383423
accuracy_score :  0.8514
걸린시간 :  782.98 초

rlr
loss : 2.30263614654541
acc : 0.10000000149011612
accuracy_score :  0.1
걸린시간 :  619.09 초


























'''