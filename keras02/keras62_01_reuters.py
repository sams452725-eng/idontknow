from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import time

(x_train,y_train),(x_test,y_test)= reuters.load_data(
    num_words=1000,    # 단어사전의 개수, 많이 사용하는 순서대로 1000개를 뽑겠다.
    # maxlen = 100,    # 문장의 최대 길이를 100으로 설정하겠다.
    test_split=0.2,    # 전체 데이터의 20%를 테스트 데이터로 사용하겠다.
)

# print(x_train)
# print(x_train.shape, y_train.shape)        #(8982,) (8982,)
# print(x_test.shape, y_test.shape)          #(2246,) (2246,)
# print(y_train)     #[ 3  4  3 ... 25  3 25]
# print(np.unique(y_train))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
# print(type(x_train))        #<class 'numpy.ndarray'>
# print(type(x_train[0]))     #<class 'list'>
# print(len(x_train[0])), print(len(x_train[1]))      #87, 56

# print('뉴스기사의 최대길이 :', max(len(i) for i in x_train))            #2376
# print('뉴스기사의 최소길이 :', min(len(i) for i in x_train))            #13
# print('뉴스기사의 평균길이 :', sum(map(len, x_train))/len(x_train))     #145.5398574927633

# 전처리(패드 시퀸스)
# y값 ohe

x_train = pad_sequences(x_train,                          
                        padding='pre',
                        maxlen=150,
                        truncating='post'
)

x_test = pad_sequences(x_test,                          
                        padding='pre',
                        maxlen=150,
                        truncating='post'
)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1) 
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(x_train.shape, x_test.shape)     #(8982, 150) (2246, 150)
# print(y_train.shape, y_test.shape)     #(8982, 46) (2246, 46)

#2. 모델구성
model = Sequential()
model.add(Embedding(450, 150))
model.add(LSTM(16, activation='relu', return_sequences=True))
model.add(LSTM(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))
# model.summary()

# exit()
#3. 컴파일,훈련
from tensorflow.keras.optimizers import Adam
# learning_rate = 0.005
learning_rate = 0.005

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=30,
        restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=10,
    verbose=1,
    factor=0.25,        
)  
import datetime
date = datetime.datetime.now()    

date = date.strftime('%m%d_%H%M')

path = './_save/reuters/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'     
filepath = ''.join([path, 'reuters', date,'_', filename])


mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    save_best_only=True,
    filepath = filepath,             
    verbose=1,
)

start_time = time.time()                      
model.fit(x_train, y_train, 
        epochs = 1000,
        batch_size = 64,
        callbacks=[es,mcp,rlr],
        validation_split=0.2,
        verbose=1,
)
end_time = time.time()



#4. 평가,예측
result = model.evaluate(x_test, y_test,)
print('loss :', result[0])
print('acc :', round(result[1], 2))

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

# exit()
accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score :', accuracy_score)
print('걸린시간 :', round(end_time - start_time, 2), '초')



### acc 0.67 이상!! ###

'''
1차
loss : 1.6948368549346924
acc : 0.59
acc_score : 0.5930543187889582
걸린시간 : 961.6 초

2차
loss : 2.1644363403320312
acc : 0.45
acc_score : 0.4452359750667854
걸린시간 : 1329.03 초

3차
loss : 2.25850248336792
acc : 0.38
acc_score : 0.38245770258236866
걸린시간 : 949.74 초

4차
loss : 1.7525197267532349
acc : 0.58
acc_score : 0.5841495992876224
걸린시간 : 1855.88 초

5차
loss : 1.9265751838684082
acc : 0.53
acc_score : 0.5280498664292075
걸린시간 : 202.74 초

6차
loss : 1.4348409175872803
acc : 0.67
acc_score : 0.6669634906500446
걸린시간 : 186.75 초

7차
loss : 2.07969069480896
acc : 0.5
acc_score : 0.4977738201246661
걸린시간 : 3229.31 초

8차
loss : 1.4811359643936157
acc : 0.65
acc_score : 0.649154051647373
걸린시간 : 2083.4 초











'''