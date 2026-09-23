import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar100
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
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
###################### 여기부터 증폭 ######################
datagen = ImageDataGenerator(
    rescale=1./255,         # 1.은 부동소수점 형변환 1/255. 해도 된다. 어디든 .을 붙이면 된다.--------------->keras43에서 20,21행의 rescale과 같은거다.
    horizontal_flip=True,   # 수평(좌우) 뒤집기 (32x32 해상도 손실 없이 효과 극대화)
    # vertical_flip=True,     # 수직(상하) 뒤집기    true는 한다, false는 안한다.
    width_shift_range=0.06,  # 32x32에서 1~2픽셀 미세 이동 (과도한 왜곡 방지)
    height_shift_range=0.06, # 32x32에서 1~2픽셀 미세 이동
    rotation_range=8,       # 8도 미세 회전 (사물 윤곽선 및 선명도 보존)
    # zoom_range=0.5,           
    # shear_range=0.7,        # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동(한마디로 찌부)
    fill_mode='nearest'     # 데이터가 이동 및 증폭하면 데이터가 이동한 쪽은 소실되고 그전에 있던 곳은 비워지게 된다. 비워진 곳은 새롭게 채워야 하는데 그때 근처에 있는 수치로 채운다.
)

augment_size=40000

randidx = np.random.choice(x_train.shape[0], size=augment_size, replace=False)     

x_augmented = x_train[randidx].copy()        
y_augmented = y_train[randidx].copy()


x_augmented = x_augmented.reshape(                              
        x_augmented.shape[0],
        x_augmented.shape[1],
        x_augmented.shape[2], 
        3)

x_augmented = datagen.flow(
                x_augmented, y_augmented,
                batch_size=augment_size,
                shuffle=False,
).next()[0]

# print(x_augmented.shape)       #(40000, 32, 32, 3)

# print(x_train.shape)        #(50000, 32, 32, 3)

# print(x_test.shape)       #(10000, 32, 32, 3)

# exit()
x_train = x_train/255.
x_test = x_test/255.


x_train = np.concatenate((x_train, x_augmented))          
y_train = np.concatenate((y_train, y_augmented))

# ★ 핵심: 뒤에 붙은 증폭 데이터를 골고루 섞어주기 (validation_split=0.2 왜곡 방지)
shuffle_idx = np.random.permutation(len(x_train))
x_train = x_train[shuffle_idx]
y_train = y_train[shuffle_idx]


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
# y_train = y_train.reshape(60000, 1)
y_train = y_train.reshape(-1, 1) 
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

# print(y_train.shape, y_test.shape)  # (50000, 100) (10000, 100)


# exit()
#2. 모델구성
input1 = Input(shape=(32, 32, 3))

# Block 1 (32x32 -> 16x16)
conv1_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input1)
conv1_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1_1)
pool1   = MaxPooling2D()(conv1_2)
drop1   = Dropout(0.2)(pool1)

# Block 2 (16x16 -> 8x8)
conv2_1 = Conv2D(128, (3, 3), padding='same', activation='relu')(drop1)
conv2_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv2_1)
pool2   = MaxPooling2D()(conv2_2)
drop2   = Dropout(0.25)(pool2)

# Block 3 (8x8 -> 4x4)
conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu')(drop2)
conv3_2 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv3_1)
pool3   = MaxPooling2D()(conv3_2)
drop3   = Dropout(0.3)(pool3)

# Block 4 (4x4 -> 2x2) - 100개 클래스를 위한 512 채널 확장
conv4_1 = Conv2D(512, (3, 3), padding='same', activation='relu')(drop3)
conv4_2 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv4_1)
pool4   = MaxPooling2D()(conv4_2)
drop4   = Dropout(0.3)(pool4)

# GAP 및 Dense 계층
gap     = GlobalAveragePooling2D()(drop4)
dense1  = Dense(512, activation='relu')(gap)
drop5   = Dropout(0.4)(dense1)
dense2  = Dense(256, activation='relu')(drop5)
drop6   = Dropout(0.3)(dense2)

output1 = Dense(100, activation='softmax')(drop6)
model   = Model(inputs=input1, outputs=output1)



# exit()
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=18,
        restore_best_weights=True,
)

start_time=time.time()
hist = model.fit(x_train, y_train, epochs=150, batch_size=128,
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




# .4 이상

'''
keras43 best
loss : 1.9274054765701294
acc : 0.49390000104904175
accuracy_score :  0.4939
걸린시간 :  228.53 초

1차
loss : 1.8821849822998047
acc : 0.5252000093460083
accuracy_score :  0.5252
걸린시간 :  705.69 초



















'''