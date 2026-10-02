from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import time

(x_train,y_train),(x_test,y_test)= imdb.load_data(
    num_words=1000,
)
# keras62-1과는 다르게 이미 train과 test의 수를 5:5로 나눠놔서 test_split을 적용할수가 없다.
# print(x_train)
# print(x_train.shape, y_train.shape)        #(25000,) (25000,)
# print(x_test.shape, y_test.shape)          #(25000,) (25000,)
# print(y_train)     #[1 0 0 ... 0 1 0]

# exit()
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

# print(x_train.shape, x_test.shape)     #(25000, 150) (25000, 150)
# print(y_train.shape, y_test.shape)     #(25000, 2) (25000, 2)


# exit()
#2. 모델구성
model = Sequential()
model.add(Embedding(1000, 100))
model.add(LSTM(16, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))



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
    factor=0.5,        
)  
import datetime
date = datetime.datetime.now()    

date = date.strftime('%m%d_%H%M')

path = './_save/imdb/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'     
filepath = ''.join([path, 'imdb', date,'_', filename])


mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    save_best_only=True,
    filepath = filepath,             
    verbose=1,
)

start_time = time.time()                      
model.fit(x_train, y_train, 
        epochs = 150,
        batch_size = 128,
        callbacks=[es,mcp,rlr],
        validation_split=0.2,

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


### acc 0.6 이상!! ###

'''
1차
loss : 187516764160.0
acc : 0.61
acc_score : 0.61496
걸린시간 : 1519.13 초

2차
loss : 0.6471636891365051
acc : 0.64
acc_score : 0.63544
걸린시간 : 1580.43 초

3차
loss : 0.5465447902679443
acc : 0.74
acc_score : 0.74292
걸린시간 : 3341.86 초



































'''