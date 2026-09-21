# 44-1 카피
import numpy as np
from keras.preprocessing.image import ImageDataGenerator      

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D, Input
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint




#1. 데이터
start_time1 = time.time()            
train_datagen = ImageDataGenerator(
    rescale=1./255,         
    # horizontal_flip=True,   
    # vertical_flip=True,     
    # width_shift_range=0.1,  
    # height_shift_range=0.1, 
    # rotation_range=5,       
    # zoom_range=1.2,           
    # shear_range=0.7,        
    # fill_mode='nearest'     
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = './_data/image/cat_dog/training_set/'
path_test = './_data/image/cat_dog/test_set/'

xy_train = train_datagen.flow_from_directory(
    path_train,     
    target_size=(150,150),           # 임의로 넣을수있지만 원데이터 형태로 하기 위해서 150으로 한다.
    batch_size=8005,
    class_mode='binary',        
    color_mode='rgb',   
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(150,150),
    batch_size=2023,
    class_mode='binary',        
    color_mode='rgb',    
)

# print(xy_train[0][0].shape)       #(8005, 150, 150, 3)   
# print(xy_train[0][1].shape)       #(8005,)   

# exit()
x_train = xy_train[0][0] 
y_train = xy_train[0][1] 
x_test = xy_test[0][0] 
y_test = xy_test[0][1] 
end_time1 = time.time()  



# print(x_train.shape, y_train.shape)        #(160, 150, 150, 1) (150,)
# print(x_test.shape, y_test.shape)          #(160, 150, 150, 1) (150,)


# exit()
#2. 모델구성
model = Sequential() 

input1  = Input(shape=(150, 150, 3))

conv1   = Conv2D(32, (3, 3), padding='same', activation='relu')(input1)
pool1   = MaxPool2D()(conv1)
drop1   = Dropout(0.2)(pool1)

conv2   = Conv2D(64, (3, 3), padding='same', activation='relu')(drop1)
pool2   = MaxPool2D()(conv2)
drop2   = Dropout(0.25)(pool2)

conv3   = Conv2D(128, (3, 3), padding='same', activation='relu')(drop2)
pool3   = MaxPool2D()(conv3)
drop3   = Dropout(0.3)(pool3)

gap     = GlobalAveragePooling2D()(drop3)
# flt   = Flatten()(drop3)
dense1  = Dense(128, activation='relu')(gap)
drop4   = Dropout(0.5)(dense1)
output1 = Dense(1, activation='sigmoid')(drop4)
model   = Model(inputs=input1, outputs=output1)
# model.summary()

# exit()
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'],
)        
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=20,
        restore_best_weights=True,
)

import datetime
date = datetime.datetime.now()    

date = date.strftime('%m%d_%H%M')

path = './_save/kaggle_CatDog/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'     
filepath = ''.join([path, 'CatDog_', date,'_', filename])


mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    save_best_only=True,
    filepath = filepath,             
    verbose=1,
)
start_time2 = time.time()            
hist = model.fit(x_train, y_train, 
            epochs = 1000,
            batch_size = 32,
            verbose=1,
            callbacks=[es,mcp],
            validation_split=0.2,
)
end_time2 = time.time()  



#4. 평가,예측
print('================= model.evaluate =================')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1], 4))

y_predict = model.predict(x_test)

y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간1 : ', round(end_time1 - start_time1, 2), '초')
print('걸린시간2 : ', round(end_time2 - start_time2, 2), '초')



# 0.77 이상

'''
1차
loss : 0.6931610703468323
acc : 0.5002
accuracy_score :  0.5002471576866041
걸린시간 :  1268.75 초

2차
loss : 0.6931503415107727
acc : 0.4998
accuracy_score :  0.49975284231339595
걸린시간 :  2068.69 초

3차
loss : 0.3654848337173462
acc : 0.8453
accuracy_score :  0.8452792881858626
걸린시간 :  503.04 초

4차
loss : 0.3819807171821594
acc : 0.8374
accuracy_score :  0.8373702422145328
걸린시간 :  373.36 초

5차





'''