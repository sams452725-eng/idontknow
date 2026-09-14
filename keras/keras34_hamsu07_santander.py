# keras33-7 카피



import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout,Input
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


'''
from sklearn.datasets import load_diabetes
datasets = load_diabetes()       
x = datasets.data
y = datasets.target
----------------------> datasets는 sklearn에서만 쓰이기 때문에 실무 데이터에선 다루지 않는다. 때문에 할 때마다 하면된다.
'''


# path = './_data/kaggle_santander/'
path = 'c:/study/_data/kaggle_santander/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

print(train_csv.shape)           #(200000, 201)
print(test_csv.shape)            #(200000, 200)    -----------> 뒤에 200이 input_dim에 들어가는 숫자다.
print(submission_csv.shape)      #(200000, 1)

# print(train_csv.info())
print(train_csv.isna().sum())          #결측치 확인
print(test_csv.isnull().sum())         #결측치 확인

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
print(x.shape, y.shape)           # (200000, 200) (200000,)

print(np.unique(y, return_counts=True))           #(array([0, 1]), array([179902,  20098]))


x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.8, random_state=337, stratify=y,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler             #preprocessing 이건 전처리다. OneHot에서 써봤다.
# scaler = MinMaxScaler()                                    #내가 받은 데이터가 무조건 쓰레기라고 상정을 해야된다. 내가 받은 데이터가 전처리가 안된 데이터가 대부분일꺼기 때문이다.
# scaler = StandardScaler()
scaler = MaxAbsScaler()                                      
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


print(np.min(x_train), np.max(x_train))      
print(np.min(x_test), np.max(x_test))        


# model = Sequential()
# model.add(Dense(10, input_dim=200, activation='relu'))
# model.add(Dropout(.2))
# model.add(Dense(10))
# model.add(Dropout(.3))
# model.add(Dense(10))
# model.add(Dropout(.5))
# model.add(Dense(10))
# model.add(Dense(1, activation='sigmoid'))  


input1 = Input(shape=(200,))     
dense1 = Dense(10, activation='relu')(input1)            
drop1 = Dropout(.2)(dense1)
dense2 = Dense(10,)(drop1)         
drop2 = Dropout(.3)(dense2)
dense3 = Dense(10,)(drop2)         
drop3 = Dropout(.5)(dense3)
dense4 = Dense(10,)(drop3)         
output1 = Dense(1, activation='sigmoid')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()




from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'],
)
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


loss = model.evaluate(x_test, y_test)
print("================= history =======================")
print('loss :', loss[0])
print('acc :', round(loss[1], 4))
print("================= history =======================")




y_pred = model.predict(x_test)          
y_pred = np.round(y_pred)           # 원칙적으로는 round 처리를 하면 안된다.



from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)

print('acc_score :', acc_score)

'''
loss : 0.23481574654579163
acc : 0.9144
acc_score : 0.9144

dropout
loss : 0.24318088591098785
acc : 0.8995
acc_score : 0.8995

함수형
loss : 0.249352365732193
acc : 0.8995
acc_score : 0.8995
'''