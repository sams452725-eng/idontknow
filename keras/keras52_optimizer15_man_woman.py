# 여자만 증폭시켜서 값을 뽑아라

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import time
from keras.preprocessing.image import ImageDataGenerator      
from sklearn.metrics import accuracy_score




#1. 데이터
np_path = './_data/men_women_npy/'
x_train = np.load(np_path + 'men_women_x_train.npy')
y_train = np.load(np_path + 'men_women_y_train.npy')
x_test = np.load(np_path + 'men_women_x_test.npy')
y_test = np.load(np_path + 'men_women_y_test.npy')
# print(x_train.shape, x_test.shape)     #(21733, 100, 100, 3) (5434, 100, 100, 3)
# print(y_train.shape, y_test)      #(21733,) [0. 1. 1. ... 0. 0. 0.]

x_train_woman = x_train[np.where(y_train> 0.0)]
y_train_woman = y_train[np.where(y_train> 0.0)]
# print(x_train_woman.shape, y_train_woman.shape)         #(7567, 100, 100, 3) (7567,)
# print(np.unique(y_train_woman, return_counts=True))     #(array([1.], dtype=float32), array([7567], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, 
    test_size=0.2, 
    shuffle=True,      
    random_state=42,
)

###################### 여기부터 증폭 ######################
train_datagen = ImageDataGenerator(
    rescale=1./255,         
    horizontal_flip=True,   
    # vertical_flip=True,     
    width_shift_range=0.1,  
    # height_shift_range=0.1, 
    rotation_range=15,      
    # zoom_range=0.5,           
    # shear_range=0.7,        
    fill_mode='nearest'     
)

augment_size=8000


# randidx = np.random.choice(x_train_woman.shape[0], size=augment_size, replace=False)       
randidx = np.random.randint(x_train_woman.shape[0], size=augment_size)     
# print(randidx)
# print(len(randidx))   #8000
# print(np.min(randidx), np.max(randidx))       # 3 7566


x_augmented = x_train_woman[randidx].copy()        
y_augmented = y_train_woman[randidx].copy()

# print(x_augmented.shape, y_augmented.shape)  # (8000, 100, 100, 3) (8000,)



# exit()
x_augmented = x_augmented.reshape(                               #(40000, 28,28,1)
        x_augmented.shape[0],
        x_augmented.shape[1],
        x_augmented.shape[2], 3)                              
# print(x_augmented.shape)       # (8000, 100, 100, 3)

x_augmented = train_datagen.flow(
                x_augmented, y_augmented,
                batch_size=augment_size,
                shuffle=False,
).next()[0]

###### 변환완료 ######
# print(x_augmented.shape)       # (8000, 100, 100, 3)

# print(x_train.shape)        # (17386, 100, 100, 3)
# x_train = x_train/255.
# x_test = x_test/255.



# exit()
x_train = np.concatenate((x_train, x_augmented))          #concatenate 정말 많이 쓴다./사슬처럼 엮다라는 의미로 다 차원의 데이터를 엮어준다.
y_train = np.concatenate((y_train, y_augmented))
# print(x_train.shape, y_train.shape)           #(25386, 100, 100, 3) (25386,)
# print(x_test.shape, y_test.shape)             #(4347, 100, 100, 3) (4347,)

# print(np.unique(y_train, return_counts=True)) #(array([0., 1.], dtype=float32), array([11361, 14025], dtype=int64))


#2. 모델
model = Sequential() 

input   = Input(shape=(100, 100, 3))

conv1_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(input)
conv1_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1_1)
pool1   = MaxPool2D()(conv1_2)
# drop1   = Dropout(0.2)(pool1)

conv2_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
conv2_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2_1)
pool2   = MaxPool2D()(conv2_2)
# drop2   = Dropout(0.25)(pool2)

conv3_1 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool2)
conv3_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv3_1)
pool3   = MaxPool2D()(conv3_2)
# drop3   = Dropout(0.3)(pool3)

conv4   = Conv2D(256, (3, 3), padding='same', activation='relu')(pool3)
pool4   = MaxPool2D()(conv4)
# drop4   = Dropout(0.3)(pool4)

gap     = GlobalAveragePooling2D()(conv4)
dense1  = Dense(128, activation='relu')(gap)
drop5   = Dropout(0.4)(dense1)
output = Dense(1, activation='sigmoid')(drop5)
model   = Model(inputs=input, outputs=output)
# model.summary()




# exit()
from tensorflow.keras.optimizers import Adam
learning_rate = 0.005
# learning_rate = 0.009


from tensorflow.keras.callbacks import ReduceLROnPlateau


model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'],
)        
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=15,
        restore_best_weights=True,
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5,        #LeraningRate를 0.5를 곱한만큼 줄어든다./ Adam은 자체 LR조절 알고리즘이 있지만 ReduceLROnPlateau와는 다른 알고리즘이기 때문에 상호보완적이다.
)
start_time = time.time()                      
hist = model.fit(x_train, y_train, 
            epochs = 150,
            batch_size = 64,                
            verbose=1,
            callbacks=[es,rlr],
            validation_split=0.5,
            shuffle=True,
)
end_time = time.time()  

#3. 평가,예측
print('================= model.evaluate =================')
loss = model.evaluate(x_test, y_test,)
print('loss :', loss[0])
print('acc :', round(loss[1], 4))

y_predict = model.predict(x_test)

y_predict = np.round(y_predict,)


acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 :', round(end_time - start_time, 2), '초')




'''
keras47 best
loss : 0.17734159529209137
acc : 0.9284
accuracy_score :  0.9284136915715863
걸린시간 : 120.21 초

1차
loss : 0.6714807748794556
acc : 0.6453
accuracy_score :  0.6452726017943409
걸린시간 : 424.13 초

2차
loss : 0.3292253911495209
acc : 0.9369
accuracy_score :  0.9368789105631211
걸린시간 : 1223.85 초


optimizer = learning_rate = 0.005
loss : 0.28238219022750854
acc : 0.9013
accuracy_score :  0.9013112491373361
걸린시간 : 1645.23 초


optimizer = learning_rate = 0.009
loss : 0.39548373222351074
acc : 0.9195
accuracy_score :  0.9194847020933977
걸린시간 : 1795.5 초

















'''