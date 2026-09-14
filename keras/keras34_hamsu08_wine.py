# keras33-8 카피

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras. layers import Dense,Dropout,Input
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping


############### acc >=.95 ###############


#1. 데이터
datasets = load_wine()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets['target']
# print(x.shape, y.shape)        #(178, 13) (178,)
# print(y)        #(178,)
# print(np.unique(y, return_counts=True))       #(array([0, 1, 2]), array([59, 71, 48]))
# print(pd.Categorical(y))


from tensorflow.keras.utils import to_categorical
y=to_categorical(y)
# print(y,y.shape)        # (178, 3)


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=.8,
    shuffle=True,
    random_state=338,
    stratify=y,
)

# print(x_train.shape, x_test.shape)          # (142, 13) (36, 13)
# print(y_train.shape, y_test.shape)          # (142, 3) (36, 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler             #preprocessing 이건 전처리다. OneHot에서 써봤다.
# scaler = MinMaxScaler()                                    #내가 받은 데이터가 무조건 쓰레기라고 상정을 해야된다. 내가 받은 데이터가 전처리가 안된 데이터가 대부분일꺼기 때문이다.
# scaler = StandardScaler()
scaler = MaxAbsScaler()                                      
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


print(np.min(x_train), np.max(x_train))      # 0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test))  


#2. 모델구성
# model = Sequential()
# model.add(Dense(10, input_dim=13, activation='relu'))
# model.add(Dropout(.2))
# model.add(Dense(10))
# model.add(Dropout(.3))
# model.add(Dense(10))
# model.add(Dropout(.5))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))   


input1 = Input(shape=(13,))     
dense1 = Dense(10, activation='relu')(input1)           
drop1 = Dropout(.2)(dense1)
dense2 = Dense(10,)(drop1)         
drop2 = Dropout(.3)(dense2)
dense3 = Dense(10,)(drop2)         
drop3 = Dropout(.5)(dense3)
dense4 = Dense(10,)(drop3)         
output1 = Dense(3, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()






from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'],)     
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=30,
                   restore_best_weights=True,
                   verbose=1,
                   )



# mcp = ModelCheckpoint(
#     monitor='val_loss', 
#     mode='auto',
#     save_best_only=True,
#     filepath = filepath,             # filepath를 똑같이 써도되냐? 된다. why? 앞은 파라미터 뒤는 변수의 형식이다. 오히려 가독성을 위해서 동일하게 한다.
#     verbose=1,
# )


start_time = time.time()                
hist = model.fit(x_train, y_train,
                 epochs = 1000, batch_size = 32, validation_split=0.2,
                 callbacks=[es,],
                 verbose=1,
                 )
end_time = time.time()    


#4. 평가,예측
result = model.evaluate(x_test, y_test,)
print('loss :', result[0])
print('acc :', round(result[1], 2))

y_predict = model.predict(x_test)
# print(y_predict)
y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)  
y_test = np.argmax(y_test, axis=1)
# print(y_test)     

# exit()
accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score :', accuracy_score)
# print('걸린시간 :', round(end_time - start_time, 2), '초')

'''
loss : 0.08506418019533157
acc : 0.94
acc_score : 0.9444444444444444

dropout
loss : 0.06122135370969772
acc : 0.97
acc_score : 0.9722222222222222

함수형
loss : 0.2700256109237671
acc : 0.92
acc_score : 0.9166666666666666
'''