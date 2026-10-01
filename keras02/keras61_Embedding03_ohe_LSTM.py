import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밋네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다'
]
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])
# print(labels.shape)    #(15,)

x_pred = ['개똥이 잘생겼다']


# exit()
token = Tokenizer()
token.fit_on_texts(docs)

x = token.texts_to_sequences(docs)
x_pred = token.texts_to_sequences(x_pred)
# print(x)
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]
# print(x_pred) 
# [[24, 27]]

# exit()
y = labels
######################## 패딩 ########################
from tensorflow.keras.preprocessing.sequence import pad_sequences
x = pad_sequences(x, padding='pre', maxlen=5, truncating='post')
x_pred = pad_sequences(x_pred, padding='pre', maxlen=5, truncating='post')
# print(x_pred.shape)      # (1, 5)
# print(x.shape)           # (15, 5)
# print(y.shape)           # (15,)

# exit()
x = x.reshape(x.shape[0], x.shape[1], 1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)

# print(x.shape, x_pred.shape)       #(15, 5, 1) (1, 5, 1)


# exit()
# from tensorflow.keras.utils import to_categorical 
# import numpy as np
# x = np.array(x).reshape(-1)           
# x = to_categorical(x)
# x = x[:, 1:]

# x_pred = np.array(x_pred).reshape(-1)           
# x_pred = to_categorical(x_pred)
# x_pred = x_pred[:, 1:]
# print(x.shape, x_pred.shape)    #(75, 30) (5, 27)



from sklearn.preprocessing import OneHotEncoder
x = np.array(x).reshape(-1,1)
x_pred = np.array(x_pred).reshape(-1,1)
ohe = OneHotEncoder(sparse_output=False)
x = ohe.fit_transform(x)
x_pred = ohe.transform(x_pred)

x = x[:, 1:].reshape(15, 5, -1)
x_pred = x_pred[:, 1:].reshape(1, 5, -1)



print(x.shape, x_pred.shape)    # (15, 5, 30) (1, 5, 30)

# exit()
x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.8, random_state=333)

#2. 모델구성
model = Sequential()
model.add(LSTM(32,input_shape=(5,30), return_sequences=True, activation='relu'))            
model.add(LSTM(16, activation='relu', return_sequences=True))
model.add(LSTM(8, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='linear'))
model.add(Dense(1, activation='sigmoid'))  


#3. 컴파일,훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'],
)
hist = model.fit(x_train, y_train, 
            epochs = 100,
            batch_size = 1,
            verbose=1,
)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("================= history =======================")
print('loss :', loss[0])
print('acc :', round(loss[1], 4))
print("================= history =======================")

y_pred = model.predict(x_test)          
y_pred = np.round(y_pred)

print(x_pred.shape)
# exit()
y_pred_result = model.predict(x_pred)

acc_score = accuracy_score(y_test, y_pred)

print('acc_score :', acc_score)
print("예측 확률값 :", y_pred_result)

'''
1차
loss : 3.8545920848846436
acc : 0.8
acc_score : 0.8
예측 확률값 : [[0.03595616]]

2차
loss : 0.3394660949707031
acc : 1.0
acc_score : 1.0
예측 확률값 : [[0.23159306]]

3차
loss : 6.263578414916992
acc : 0.6667
acc_score : 0.6666666666666666
예측 확률값 : [[0.2915403]]

4차
loss : 0.0009151453268714249
acc : 1.0
acc_score : 1.0
예측 확률값 : [[0.04966995]]

5차
loss : 2.9807223356215218e-09
acc : 1.0
acc_score : 1.0
예측 확률값 : [[0.99997926]]

ohe 작업
1차
loss : 4.241973400115967
acc : 0.6667
acc_score : 0.6666666666666666
예측 확률값 : [[0.00012727]]








'''