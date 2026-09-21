import numpy as np
from keras.preprocessing.image import ImageDataGenerator      

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D, Input
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint




#1. 데이터
np_path = './_data/men_women_npy/'
x_train = np.load(np_path + 'men_women_x_train.npy')
y_train = np.load(np_path + 'men_women_y_train.npy')
x_test = np.load(np_path + 'men_women_x_test.npy')
y_test = np.load(np_path + 'men_women_y_test.npy')

#2. 모델구성
model = Sequential() 

input   = Input(shape=(100, 100, 3))

conv1   = Conv2D(32, (3, 3), padding='same', activation='relu')(input)
pool1   = MaxPool2D()(conv1)

conv2   = Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
pool2   = MaxPool2D()(conv2)

conv3   = Conv2D(128, (3, 3), padding='same', activation='relu')(pool2)
pool3   = MaxPool2D()(conv3)

conv4   = Conv2D(256, (3, 3), padding='same', activation='relu')(pool3)
pool4   = MaxPool2D()(conv4)

gap     = GlobalAveragePooling2D()(pool4)
# flt   = Flatten()(pool4)
dense1  = Dense(256, activation='relu')(gap)
drop    = Dropout(0.5)(dense1)
output  = Dense(1, activation='sigmoid')(drop)
model   = Model(inputs=input, outputs=output)
# model.summary()




# exit()
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'],
)        
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        restore_best_weights=True,
)

import datetime
date = datetime.datetime.now()    

date = date.strftime('%m%d_%H%M')

path = './_save/men_women/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'     
filepath = ''.join([path, 'mw_', date,'_', filename])


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
            # batch_size = 16,                #imagegenerate를 사용하면 이미 batch_size가 적용이 되어있기 때문에 model.fit에서는 필요없다.
            verbose=1,
            callbacks=[es,mcp],
            validation_split=0.2,
)
end_time = time.time()  



# exit()
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







# .94 or .95 이상
'''
1차
loss : 0.44239670038223267
acc : 0.8571
accuracy_score :  0.8571428571428571
걸린시간 : 3.73 초

2차
loss : 0.4278835654258728
acc : 0.8571
accuracy_score :  0.8571428571428571
걸린시간 : 3.92 초

3차
loss : 0.25425511598587036
acc : 0.8571
accuracy_score :  0.8571428571428571
걸린시간 : 6.0 초

4차
loss : 0.34461069107055664
acc : 0.8571
accuracy_score :  0.8571428571428571
걸린시간 : 5.72 초

5차
loss : 0.6472619771957397
acc : 0.6923
accuracy_score :  0.6923076923076923
걸린시간 : 3.44 초

6차
loss : 0.6396911144256592
acc : 0.6154
accuracy_score :  0.6153846153846154
걸린시간 : 4.92 초

7차
loss : 0.5591691136360168
acc : 0.6154
accuracy_score :  0.6153846153846154
걸린시간 : 7.09 초

8차
loss : 0.6181853413581848
acc : 0.6154
accuracy_score :  0.6153846153846154
걸린시간 : 7.55 초

9차
loss : 0.3588828444480896
acc : 0.8462
accuracy_score :  0.8461538461538461
걸린시간 : 7.66 초

10차
loss : 0.39589375257492065
acc : 0.8462
accuracy_score :  0.8461538461538461
걸린시간 : 6.82 초

11차
loss : 0.5824876427650452
acc : 0.6154
accuracy_score :  0.6153846153846154
걸린시간 : 5.28 초

12차
loss : 0.680862307548523
acc : 0.6154
accuracy_score :  0.6153846153846154
걸린시간 : 4.86 초

13차
loss : 0.4086509048938751
acc : 0.8462
accuracy_score :  0.8461538461538461
걸린시간 : 7.1 초

14차
loss : 0.5216621160507202
acc : 0.7692
accuracy_score :  0.7692307692307693
걸린시간 : 18.63 초

15차
loss : 0.5554824471473694
acc : 0.6154
accuracy_score :  0.6153846153846154
걸린시간 : 17.98 초

16차
loss : 0.4130637049674988
acc : 0.9231
accuracy_score :  0.9230769230769231
걸린시간 : 18.47 초

17차
loss : 0.590761661529541
acc : 0.6154
accuracy_score :  0.6153846153846154
걸린시간 : 18.55 초

18차
loss : 0.5831642746925354
acc : 0.6154
accuracy_score :  0.6153846153846154
걸린시간 : 17.68 초

19차
loss : 0.5744005441665649
acc : 0.6154
accuracy_score :  0.6153846153846154
걸린시간 : 17.88 초

20차
loss : 0.6179365515708923
acc : 0.6154
accuracy_score :  0.6153846153846154
걸린시간 : 17.91 초

21차
loss : 0.5921026468276978
acc : 0.6154
accuracy_score :  0.6153846153846154
걸린시간 : 17.78 초

22차
loss : 0.6714149117469788
acc : 0.6154
accuracy_score :  0.6153846153846154
걸린시간 : 18.21 초

23차
loss : 0.5843072533607483
acc : 0.6154
accuracy_score :  0.6153846153846154
걸린시간 : 20.52 초

24차
loss : 0.6725174784660339
acc : 0.6154
accuracy_score :  0.6153846153846154
걸린시간 : 21.21 초

25차
loss : 0.5827531218528748
acc : 0.7692
accuracy_score :  0.7692307692307693
걸린시간 : 18.57 초

26차
loss : 0.6459460854530334
acc : 0.6154
accuracy_score :  0.6153846153846154
걸린시간 : 16.66 초

27차
loss : 0.207860067486763
acc : 0.9122
accuracy_score :  0.9122193595877807
걸린시간 : 165.77 초

28차
loss : 0.18998779356479645
acc : 0.9223
accuracy_score :  0.9223408170776591
걸린시간 : 116.82 초

29차
loss : 0.17734159529209137
acc : 0.9284
accuracy_score :  0.9284136915715863
걸린시간 : 120.21 초

30차
loss : 0.1925615668296814
acc : 0.9253
accuracy_score :  0.9252852410747148
걸린시간 : 109.42 초


















'''