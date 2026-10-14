import numpy as np
from keras.preprocessing.image import ImageDataGenerator      

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D, Input
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.applications import VGG16




#1. 데이터
np_path = './_save/rps_npy/'
x_train = np.load(np_path + 'keras46_02_x_train.npy')
y_train = np.load(np_path + 'keras46_02_y_train.npy')
x_test = np.load(np_path + 'keras46_02_x_test.npy')
y_test = np.load(np_path + 'keras46_02_y_test.npy')

#2. 모델구성
model = Sequential() 

vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(250, 250, 3)
)
vgg16.trainable = True

input = Input(shape=(250, 250, 3))
vgg16 = vgg16(input)
flt = Flatten()(vgg16)
dense1 = Dense(32)(flt)
dense2 = Dense(16)(dense1)
output = Dense(3, activation='softmax')(dense1)

model = Model(inputs=input, outputs=output)


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
    mode='min',
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

vgg16
loss : 0.3461276590824127
acc : 0.8755
accuracy_score :  0.8755463538072233
걸린시간 : 2369.5 초

vgg16.trainable = True
1차
loss : 1.023339867591858
acc : 0.3976
accuracy_score :  0.3975609756097561
걸린시간 : 328.15 초

2차
loss : 1.0193079710006714
acc : 0.4512
accuracy_score :  0.45121951219512196
걸린시간 : 311.43 초

3차
loss : 0.8183647394180298
acc : 0.6463
accuracy_score :  0.6463414634146342
걸린시간 : 326.32 초




'''

'''
전이학습
Transfer Learning

Feature Extraction(대충 가중치를 가져다가 쓰는거로 이해하면 된다.)
    사전 학습된 모델을 고정된 특징 추출기로 활용

Pre-trained Model
사전학습 모델

사전학습 모델 기반의 파인튜닝(Fine-Tuning)

어쩌구 저쩌구... 이제 위에 용어들이 뭔말인지 알것죠?     예!!
'''