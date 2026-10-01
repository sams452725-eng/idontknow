import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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

x_predict = ['개똥이 잘생겼다']


# exit()
token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7,
# '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15,
# '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22,
# '재밋네요': 23, '개똥이': 24, '바보': 25, '말똥이': 26, '잘생겼다': 27, '길동이': 28, '또': 29, '구라친다': 30}

# x = token.text_to_sequences(docs)
# print(x)
# AttributeError: 'Tokenizer' object has no attribute 'text_to_sequences'. Did you mean: 'texts_to_sequences'?
x_pred = token.texts_to_sequences(x_predict)
x = token.texts_to_sequences(docs)
# print(x)
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]
# 크기가 다 다르기 때문에 가장 긴 크기에 맞춰준다. 때론 너무 길때는 적당하게 짤라줄수도 있다.
y = labels
######################## 패딩 ########################
from tensorflow.keras.preprocessing.sequence import pad_sequences
x = pad_sequences(x,                          # x를 padding하겠다.
                         padding='pre',              #앞을 '0'으로 채우려면 padding='pre', 반대로 뒤를 '0'으로 채우려면 padding='post'  
                         maxlen=5,
                         truncating='post'            # 최대 숫자보다 작은 수로 잡으면 디폴트는 앞이 짤린다. 'post'로 바꾸면 뒤가 짤린다.
)
# print(x, x.shape)     # (15, 5)
# [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  0  1  5  6]
#  [ 0  0  7  8  9]
#  [10 11 12 13 14]
#  [ 0  0  0  0 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0 17 18]
#  [ 0  0  0 19 20]
#  [ 0  0  0  0 21]
#  [ 0  0  0  2 22]
#  [ 0  0  0  1 23]
#  [ 0  0  0 24 25]
#  [ 0  0  0 26 27]
#  [ 0  0 28 29 30]]
x_pred = pad_sequences(x_pred, padding='pre', maxlen=5, truncating='post')
# print(x_pred.shape)      # (1, 5)
# print(y.shape)   # (15,)


# exit()
x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.8, random_state=333)


#2. 모델구성
model = Sequential()
model.add(Dense(256, input_dim=5, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  


#3. 컴파일,훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'],
)
hist = model.fit(x_train, y_train, 
            epochs = 500,
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
loss : 0.8243463635444641
acc : 0.3333
acc_score : 0.3333333333333333
예측 확률값 : [[0.36418372]]







'''