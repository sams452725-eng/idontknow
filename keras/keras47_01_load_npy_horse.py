import numpy as np
from keras.preprocessing.image import ImageDataGenerator      

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D, Input
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint




#1. 데이터
np_path = './_data/horse-human_npy/'
x_train = np.load(np_path + 'keras46_01_x_train.npy')
y_train = np.load(np_path + 'keras46_01_y_train.npy')
x_test = np.load(np_path + 'keras46_01_x_test.npy')
y_test = np.load(np_path + 'keras46_01_y_test.npy')

#2. 모델구성
model = Sequential() 

input   = Input(shape=(250, 250, 3))

conv1   = Conv2D(32, (3, 3), padding='same', activation='relu')(input)
pool1   = MaxPool2D()(conv1)

conv2   = Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
pool2   = MaxPool2D()(conv2)

conv3   = Conv2D(128, (3, 3), padding='same', activation='relu')(pool2)
pool3   = MaxPool2D()(conv3)

# conv4   = Conv2D(256, (3, 3), padding='same', activation='relu')(pool3)
# pool4   = MaxPool2D()(conv4)

gap     = GlobalAveragePooling2D()(pool3)
# flt   = Flatten()(drop3)
dense1  = Dense(128, activation='relu')(gap)
drop    = Dropout(0.5)(dense1)
output  = Dense(2, activation='softmax')(drop)
model   = Model(inputs=input, outputs=output)


model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'],
)        
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=25,
        restore_best_weights=True,
)

import datetime
date = datetime.datetime.now()    

date = date.strftime('%m%d_%H%M')

path = './_save/horse_human/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'     
filepath = ''.join([path, 'horse_', date,'_', filename])


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
loss : 0.06537610292434692
acc : 0.9903
accuracy_score :  0.9902912621359223
걸린시간 : 152.15 초

2차
loss : 0.04313987120985985
acc : 0.9903
accuracy_score :  0.9902912621359223
걸린시간 : 557.98 초

3차
loss : 0.05150779336690903
acc : 0.9903
accuracy_score :  0.9902912621359223
걸린시간 : 182.58 초

4차
loss : 0.6924636960029602
acc : 0.5049
accuracy_score :  0.5048543689320388
걸린시간 : 35.02 초

5차
loss : 0.6933155059814453
acc : 0.5049
accuracy_score :  0.5048543689320388
걸린시간 : 78.02 초

6차
loss : 0.08479611575603485
acc : 0.9903
accuracy_score :  0.9902912621359223
걸린시간 : 102.14 초

7차
loss : 0.13522732257843018
acc : 0.9854
accuracy_score :  0.9854368932038835
걸린시간 : 134.0 초

8차
loss : 0.05058010295033455
acc : 0.9854
accuracy_score :  0.9854368932038835
걸린시간 : 79.25 초

9차
loss : 0.08648578822612762
acc : 0.9903
accuracy_score :  0.9902912621359223
걸린시간 : 126.9 초

10차
loss : 0.012464097701013088
acc : 1.0
accuracy_score :  1.0
걸린시간 : 115.09 초


















'''