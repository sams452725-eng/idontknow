# keras21 카피

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
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
model = Sequential()
model.add(Dense(30, input_dim=30,activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1, activation='sigmoid'))     # output Dense에 activation을 넣으면 무조건 'sigmoid'다.





#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
            #   metrics=['accuracy'],   ---------------> 리스트 형식이기 때문에 추후에 다른 값이 추가 될수도 있다.
              metrics=['acc'],          #----------------> accuracy가 길어서 축약해도 같다.
              )         # 2진분류에서 loss는 무조건 'binary_crossentropy'다.

es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=20,
        restore_best_weights=True,
)
start_time = time.time()           #현재시간을 반환. 즉, 시작 시간     
hist = model.fit(x_train, y_train, 
            epochs = 1000,
            batch_size = 32,
            verbose=1,
            callbacks=[es],
            validation_split=0.3,
)
end_time = time.time()             #현재시간을 반환. 즉, 끝 시간



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

# (MinMaxScaler)
# loss : 0.16521556675434113
# acc : 0.924
# acc_score : 0.9239766081871345

# (StandardScaler)
# loss : 0.15324611961841583
# acc : 0.9532
# acc_score : 0.9532163742690059

# (MaxAbsScaler)                                      
# loss : 0.12642121315002441
# acc : 0.9649
# acc_score : 0.9649122807017544
















