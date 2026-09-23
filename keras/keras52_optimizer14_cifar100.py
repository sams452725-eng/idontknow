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



from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
# y_train = y_train.reshape(60000, 1)
y_train = y_train.reshape(-1, 1) 
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape, y_test.shape)  # (50000, 100) (10000, 100)


# exit()
#2. 모델구성
model = Sequential() 
input1 = Input(shape=(32, 32, 3))

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

output1 = Dense(100, activation='softmax')(drop5)
model   = Model(inputs=input1, outputs=output1)



# exit()
#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.0001
# learning_rate = 0.009 

from tensorflow.keras.callbacks import ReduceLROnPlateau


model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate),
              metrics=['acc'])
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=12,
        restore_best_weights=True,
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5,        #LeraningRate를 0.5를 곱한만큼 줄어든다./ Adam은 자체 LR조절 알고리즘이 있지만 ReduceLROnPlateau와는 다른 알고리즘이기 때문에 상호보완적이다.
)
start_time=time.time()
hist = model.fit(x_train, y_train, epochs=150, batch_size=128,
          verbose=1,
          validation_split=.2,
          callbacks=[es,rlr]
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



optimizer = learning_rate = 0.0001
loss : 2.0595245361328125
acc : 0.5449000000953674
accuracy_score :  0.5449
걸린시간 :  2380.44 초

optimizer = learning_rate = 0.009
loss : 4.606186389923096
acc : 0.009999999776482582
accuracy_score :  0.01
걸린시간 :  1101.53 초













'''