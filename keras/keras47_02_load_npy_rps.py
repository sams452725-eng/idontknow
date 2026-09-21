import numpy as np
from keras.preprocessing.image import ImageDataGenerator      

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D, Input
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint




#1. 데이터
np_path = './_data/rps_npy/'
x_train = np.load(np_path + 'keras46_02_x_train.npy')
y_train = np.load(np_path + 'keras46_02_y_train.npy')
x_test = np.load(np_path + 'keras46_02_x_test.npy')
y_test = np.load(np_path + 'keras46_02_y_test.npy')

#2. 모델구성
model = Sequential() 

input1  = Input(shape=(250, 250, 3))

conv1   = Conv2D(32, (3, 3), padding='same', activation='relu')(input1)
pool1   = MaxPool2D()(conv1)

conv2   = Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
pool2   = MaxPool2D()(conv2)

conv3   = Conv2D(128, (3, 3), padding='same', activation='relu')(pool2)
pool3   = MaxPool2D()(conv3)

gap     = GlobalAveragePooling2D()(pool3)
# flt   = Flatten()(drop3)
dense1  = Dense(128, activation='relu')(gap)
drop4   = Dropout(0.5)(dense1)
output1 = Dense(3, activation='softmax')(drop4)
model   = Model(inputs=input1, outputs=output1)


model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'],
)        
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=15,
        restore_best_weights=True,
)

import datetime
date = datetime.datetime.now()    

date = date.strftime('%m%d_%H%M')

path = './_save/rps/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'     
filepath = ''.join([path, 'rps_', date,'_', filename])


mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    save_best_only=True,
    filepath = filepath,             
    verbose=1,
)
start_time = time.time()            
hist = model.fit(x_train, y_train, 
            epochs = 100,
            batch_size = 16,
            verbose=1,
            callbacks=[es,mcp],
            validation_split=0.2,
)
end_time = time.time()  




#3. 평가,예측
print('================= model.evaluate =================')
loss = model.evaluate(x_test, y_test,)
print('loss :', loss[0])
print('acc :', round(loss[1], 4))

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)


acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 :', round(end_time - start_time, 2), '초')


'''
1차
loss : 0.0001456231257179752
acc : 1.0
accuracy_score :  1.0
걸린시간 : 1137.48 초

2차
loss : 0.0018137601437047124
acc : 1.0
accuracy_score :  1.0
걸린시간 : 242.72 초
















'''