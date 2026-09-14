# 23-1 카피

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras. layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_iris()
# print(datasets)
# print(datasets.DESCR)      #-------------- 실무에서 .DESCR는 없다. 대신 pandas에 describe라는 함수가 있다.
# print(datasets.feature_names)         #--------------> pandas에서는 colums라는 함수가 있다.


x = datasets.data
y = datasets['target']
# print(x.shape, y.shape)            # (150, 4) (150,)
# print(y)                           # (150,)
# print(np.unique(y, return_counts=True))       #(array([0, 1, 2]), array([50, 50, 50]))         #--------------> pandas에서는 value_counts가 있다.
# print(pd.Categorical(y))



# #################### ONEHOT 1. to_categorical ####################
# from tensorflow.keras.utils import to_categorical             # (150,) 데이터를 (150, 3)로 바꾸는 작업이다.
# y = to_categorical(y)
# print(y, y.shape)
# # train_test 전에 onehot을 하는 이유는 train_test를 먼저하면 2번 작업을 해야하는 번거로움이 있다.
# '0'부터 시작하는 칼럼에 대해서는 아주 좋으나, '0'이 없이 '1'부터 시작하는 칼럼에서는 강제로 '0'이 들어간 칼럼을 만들어서 쓸데없는 연산을 하게 된다. 그럼 오히려 성능저하와 쓸데없는 자원낭비가 생긴다.

#################### ONEHOT 2. pandas ####################
# y = pd.get_dummies(y, dtype=int)
# print(y)

#################### ONEHOT 3. sklearn ####################
from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(150,1)      #(150, 1)
y = y.reshape(-1,1)      #(150, 1)        ---------------------> 늘상 y값의 범위를 알수없기 때문에 앞에 -1을 쓰면 1부터 전체 y의 범위의 제일 끝까지 범위가 설정된다.
print(y, y.shape)

# exit()
# ohe = OneHotEncoder()       # sparse라는 혼돈행렬의 형태로 값이 반환된다.
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y)
print(y, y.shape)        # (150, 3)
# reshape는 절대 1.내용(값)과 2.순서를 바꾸면 안된다. 바뀐걸 알았으면 '데이터 조작', 몰랐으면 '데이터 오염'이다.

# exit()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.8, 
    shuffle=True, 
    random_state=333, 
    stratify=y,
)
print(x_train.shape, x_test.shape)          # (120, 4) (30, 4)
print(y_train.shape, y_test.shape)          # (120, 3) (30, 3)

# exit()
#2. 모델구성
model = Sequential()
# model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, input_shape=(4,), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))    
# 'softmax'는 모든 값을 더했을 때 1이 넘지 않게 만드는 함수.(sigmoid처럼 값을 한정시키는 활성화 함수다.)

'''
원데이터 -----> input_shape
(n, 4)--->(4,)
(n,100,3)---->(100,3)
(n,100,100,3)------>(100,100,3)
'''

exit()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'],)     # 다중분류에서 loss는 'categorical_crossentropy' 하나다.
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)          # 코드에러가 뜬 이유는 각 값의 가치가 동일한데 어느 한쪽이 더 큰 가중치를 갖게되서 생긴다. 즉, 각각의 값을 백터화 시켜서 처리한다. 때문에 line32의 함수를 써줘야한다.
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=8,
          verbose=1,
          validation_split=.2,
          callbacks=[es],
          )
end_time = time.time()

#4. 평가,예측
result = model.evaluate(x_test, y_test,)
print('loss :', result[0])
print('acc :', round(result[1], 2))

y_predict = model.predict(x_test)
print(y_predict)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)  #[0 2 0 2 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 2]
y_test = np.argmax(y_test, axis=1)
print(y_test)     #[0 2 0 1 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]

# exit()
accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score :', accuracy_score)
print('걸린시간 :', round(end_time - start_time, 2), '초')
# acc_score : 0.9666666666666667
# 걸린시간 : 16.39 초