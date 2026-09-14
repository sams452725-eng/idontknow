# keras23-3 카피

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras. layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

# acc >= .93



#1. 데이터
datasets = fetch_covtype()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets['target']
# print(x.shape, y.shape)        #(581012, 54) (581012,)
# print(y)        #(581012,)     #(581012, 54) (581012,)
print(np.unique(y, return_counts=True))       #(array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))
# print(pd.Categorical(y))


#################### ONEHOT 1. to_categorical ####################
# from tensorflow.keras.utils import to_categorical
# y_oh = to_categorical(y)             
# print(y_oh[:10])
# print(y[:10])
# print(y_oh.shape, y.shape)
# train_test 전에 onehot을 하는 이유는 train_test를 먼저하면 2번 작업을 해야하는 번거로움이 있다.
# '0'부터 시작하는 칼럼에 대해서는 아주 좋으나, '0'이 없이 '1'부터 시작하는 칼럼에서는 강제로 '0'이 들어간 칼럼을 만들어서 쓸데없는 연산을 하게 된다. 그럼 오히려 성능저하와 쓸데없는 자원낭비가 생긴다.

#################### ONEHOT 2. pandas ####################
# y = pd.get_dummies(y, dtype=int)
# print(y)

#################### ONEHOT 3. sklearn ####################
from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(150,1)      #(150, 1)
y = y.reshape(-1,1)      #(150, 1)        ---------------------> 늘상 y값의 범위를 알수없기 때문에 앞에 -1을 쓰면 1부터 전체 y의 범위의 제일 끝까지 범위가 설정된다.
print(y, y.shape)

# # exit()
ohe = OneHotEncoder()       # sparse라는 혼돈행렬의 형태로 값이 반환된다.
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y)
print(y)
# reshape는 절대 1.내용(값)과 2.순서를 바꾸면 안된다. 바뀐걸 알았으면 '데이터 조작', 몰랐으면 '데이터 오염'이다.






# exit()
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=.8,
    shuffle=True,
    random_state=338,
    stratify=y,
)

# print(x_train.shape, x_test.shape)          # (464809, 54) (116203, 54)
# print(y_train.shape, y_test.shape)          # (464809, 8) (116203, 8)




from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler             #preprocessing 이건 전처리다. OneHot에서 써봤다.
# scaler = MinMaxScaler()                                    #내가 받은 데이터가 무조건 쓰레기라고 상정을 해야된다. 내가 받은 데이터가 전처리가 안된 데이터가 대부분일꺼기 때문이다.
# scaler = StandardScaler()
scaler = MaxAbsScaler()                                      
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


print(np.min(x_train), np.max(x_train))      # 0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test))        # -0.0012367054167697258 1.0

# exit()
#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=54, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='softmax'))   



from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'],)     
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=30,
                   restore_best_weights=True,
                   verbose=1,
                   )

##################### mcp 세이브 파일명 만들기 start!! (mcp에 국한된게 아니라 일반적인 파일 생성시에 편하게 작업하는 방법) #####################
import datetime
date = datetime.datetime.now()    #현재 시간 반환
# print(date)   # 2026-09-14 11:42:51.252286
# print(type(date))   # <class 'datetime.datetime'>

date = date.strftime('%m%d_%H%M')
# print(date)    # 0914_1147
# print(type(date))   # <class 'str'>

path = './_save/keras45/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'      # 컴파일을 할때 hist에서 epoch와 val_loss가 있기에 거기서 땡겨온다.
# 04d 4자리의 int 표기, .4f는 소수점 4자리까지 표기 -----> {} Dictionary 방식
filepath = ''.join([path, 'k45_', date,'_', filename])      # ''은 파일명을 일단 공백으로 만든다. if, 'aa'가 있으면 aa 뒤로 join에 적어 놓은 것들이 들어온다.
# 생성 파일명 ex) './_save/keras30/' + 'k30_' + '0914_1147'.keras

##################### mcp 세이브 파일명 만들기 finish!! #####################


# exit()

mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    save_best_only=True,
    filepath = filepath,             # filepath를 똑같이 써도되냐? 된다. why? 앞은 파라미터 뒤는 변수의 형식이다. 오히려 가독성을 위해서 동일하게 한다.
    verbose=1,
)


start_time = time.time()                
hist = model.fit(x_train, y_train,
                 epochs = 1000, batch_size = 32, validation_split=0.2,
                 callbacks=[es,mcp],
                 verbose=1,
                 )
end_time = time.time()             


#4. 평가,예측
result = model.evaluate(x_test, y_test,)
print('loss :', result[0])
print('acc :', round(result[1], 2))

y_predict = model.predict(x_test)
print(y_predict)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)  
y_test = np.argmax(y_test, axis=1)
print(y_test)     

# exit()
accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score :', accuracy_score)
print('걸린시간 :', round(end_time - start_time, 2), '초')

'''
loss : 0.4972160756587982
acc : 0.79
acc_score : 0.7904184917773207
걸린시간 : 1980.61 초
'''