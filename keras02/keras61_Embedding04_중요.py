# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
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

x_pred = token.texts_to_sequences(x_predict)
x = token.texts_to_sequences(docs)
######################## 패딩 ########################
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x,                          
                         padding='pre',
                         maxlen=5,
                         truncating='post'
)
# print(padded_x, padded_x.shape)     # (15, 5)

#2. 모델 구성
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN 

model = Sequential()
################## 임베딩1 ##################
model.add(Embedding(input_dim=30, output_dim=100, input_length=5))
#                   단어사전의 개수,    차원,
model.add(SimpleRNN(10))
model.add(Dense(1))
#  embedding (Embedding)       (None, 5, 100)            3000      
#  simple_rnn (SimpleRNN)      (None, 10)                1110

################## 임베딩2 ##################
model.add(Embedding(input_dim=30, output_dim=100))     #input_length는 굳이 명시하지 않아도 알아서 맞춰준다.
#                   단어사전의 개수,    차원,            # 차원은 output_node와 같은 개념이다. 너무 작으면 벡터화 했을 때 정보가 많이 손실될 수 있다.
model.add(SimpleRNN(10))
model.add(Dense(1))

################## 임베딩3 ##################
# model.add(Embedding(30, 100))     # 순서는 다른 layers와 다르게 앞에서부터 input_dim, output_dim이다
model.add(Embedding(30, 100, input_length=5))     #그냥 수치만 넣어주는건 앞에 2개만 가능하다. 단, input_length는 알아서 맞춰주기 때문에 굳이 넣지 않아도 된다.
model.add(SimpleRNN(10))
model.add(Dense(1))










































model.summary()
exit()
#3. 컴파일,훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'],)

model.fit(padded_x, labels, epochs = 10, verbose=1,)
