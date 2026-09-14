# keras22 카피



import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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


model = Sequential()
model.add(Dense(10, input_dim=200, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  


model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'],
)
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=300,
        restore_best_weights=True,
)
start_time = time.time()
hist = model.fit(x_train, y_train, 
            epochs = 1500,
            batch_size = 512,
            verbose=1,
            callbacks=[es],
            validation_split=0.2,
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

# (MinMaxScaler)
# loss : 0.2308207005262375
# acc : 0.915
# acc_score : 0.91495

# (StandardScaler)
# loss : 0.2377590537071228
# acc : 0.9125
# acc_score : 0.91255

# (MaxAbsScaler)                                      
# loss : 0.23489846289157867
# acc : 0.9138
# acc_score : 0.913825

# print("========================================")
# print(submission_csv)
# y_submit = model.predict(test_csv)

# submission_csv['target'] = y_submit
# print(submission_csv)
# print(submission_csv.shape)


# submission_csv.to_csv(path + 'submit/' + 'submit_0911_0950.csv')
