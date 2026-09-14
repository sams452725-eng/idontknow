# keras21 카피

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_breast_cancer          # 유방암 관련 데이터셋 불러오기

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)             # 실무에서 쓸 일이 없다. sklearn에서 제공하는 교육자료기 때문이다.
print(datasets.feature_names)

# x = datasets.data
x = datasets['data']             # history를 불러올때 dictionary를 땡겨올때 다음과 같은 형식을 썼다. 즉, dictionary의 key값을 입력하면 value값을 불러온다. datasets는 dictionary 형식이다.
y = datasets.target

print(x.shape, y.shape)          # (569, 30) (569,)
print(type(x))                   # <class 'numpy.ndarray'>                  # 파이썬의 type이라는 함수는 데이터의 형식을 보여준다.

# 범주형 모델에서는 반드시 y범주를 확인해야된다!!
# 범주형에서는 데이터 값의 array가 비슷할수록 퀄리티가 좋다.

print(y)
# 0과 1의 개수가 몇개인지 찾아보기 : numpy
print(np.unique(y))         # [0 1] -----> 즉, 2진분류
print(np.unique(y, return_counts=True))         # (array([0, 1]), array([212, 357])) --------> 0의 개수는 212개, 1의 개수는 357

# 0과 1의 개수가 몇개인지 찾아보기 : pandas
print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212

print(pd.Series(y).value_counts())
# 1    357
# 0    212
# ----------------> numpy나 pandas 둘중에 아무거나 하나 생각 나는거 쓰면 된다.


x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.7, test_size=0.3, random_state=337, stratify=y,)

print(np.unique(y_train, return_counts=True))
# (array([0, 1]), array([159, 239]))    -------------->     (array([0, 1]), array([148, 250]))   why? stratify=y를 입력하므로 퍼센트지 만큼 나눠 들어간다.
print(np.unique(y_test, return_counts=True))          #----------------> test값은 평가에만 들어가기 때문에 훈련에 영향을 끼치지 않아서 값이 불균형해도 문제가 없다.
# (array([0, 1]), array([ 53, 118]))      -------------->     (array([0, 1]), array([ 64, 107]))


print(x_train.shape, x_test.shape)        #(398, 30) (171, 30)
print(y_train.shape, y_test.shape)        #(398,) (171,)

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


#2. 모델구성
# model = Sequential()
# model.add(Dense(30, input_dim=30,activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))     # output Dense에 activation을 넣으면 무조건 'sigmoid'다.





# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam',
#               metrics=['acc'],
#               )

# es = EarlyStopping(monitor='val_loss', mode='min',
#                    patience=30,
#                    restore_best_weights=True,
#                    verbose=1,
#                    )

# ##################### mcp 세이브 파일명 만들기 start!! (mcp에 국한된게 아니라 일반적인 파일 생성시에 편하게 작업하는 방법) #####################
# import datetime
# date = datetime.datetime.now()    #현재 시간 반환
# # print(date)   # 2026-09-14 11:42:51.252286
# # print(type(date))   # <class 'datetime.datetime'>

# date = date.strftime('%m%d_%H%M')
# # print(date)    # 0914_1147
# # print(type(date))   # <class 'str'>

path = './_save/keras39/'
# filename = '{epoch:04d}-{val_loss:.4f}.keras'      # 컴파일을 할때 hist에서 epoch와 val_loss가 있기에 거기서 땡겨온다.
# # 04d 4자리의 int 표기, .4f는 소수점 4자리까지 표기 -----> {} Dictionary 방식
# filepath = ''.join([path, 'k30_', date,'_', filename])      # ''은 파일명을 일단 공백으로 만든다. if, 'aa'가 있으면 aa 뒤로 join에 적어 놓은 것들이 들어온다.
# # 생성 파일명 ex) './_save/keras30/' + 'k30_' + '0914_1147'.keras

# ##################### mcp 세이브 파일명 만들기 finish!! #####################


# # exit()

# mcp = ModelCheckpoint(
#     monitor='val_loss', 
#     mode='auto',
#     save_best_only=True,
#     filepath = filepath,             # filepath를 똑같이 써도되냐? 된다. why? 앞은 파라미터 뒤는 변수의 형식이다. 오히려 가독성을 위해서 동일하게 한다.
#     verbose=1,
# )


# start_time = time.time()                
# hist = model.fit(x_train, y_train,
#                  epochs = 1000, batch_size = 32, validation_split=0.2,
#                  callbacks=[es,mcp],
#                  verbose=1,
#                  )
# end_time = time.time()     


model = load_model(path +'k30_0914_1432_0040-0.0517.keras')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("================= history =======================")
print('loss :', loss[0])        # loss : 0.17172008752822876 ----------->loss값은 상대적인 값이라서 아무리 잘 만든 모델이라도 단 한번의 훈련으로 평가할수없다.
# loss : [0.17146167159080505, 0.9356725215911865]  ----------------> 앞에 loss값은 기존 'binary_crossentropy'으로 평가한 값이고, 뒤에 loss값은 metrics=['acc']로 평가한 값이다.
print('acc :', round(loss[1], 4))         # round는 파이썬 기초 함수 중에 반올림이다. loss[1] 뒤에 ', 4'는 소수점 4번째에서 반올림 하라는 의미다.
print("================= history =======================")

y_pred = model.predict(x_test)          
# print(y_pred[:10])
y_pred = np.round(y_pred)
# print(y_pred[:10])

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)

print('acc_score :', acc_score)

'''
loss : 0.18087345361709595
acc : 0.9298
acc_score : 0.9298245614035088
'''















