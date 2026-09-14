# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)          #[10886 rows x 11 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)           #[6493 rows x 8 columns]

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
print(submission)         #[6493 rows x 1 columns]

print(train_csv.shape)              #[6493 rows x 1 columns]
print(test_csv.shape)               #(6493, 8)
print(submission.shape)             #(6493, 1)



print(train_csv.info())          # 결측치 없음
print(test_csv.info())           # 결측치 없음

print(train_csv.describe())
############ 결측치 확인 ############
train_csv.isna().sum()
print(train_csv.isna().sum())
print(test_csv.isnull().sum())






############ x,y 분리 ############
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x, x.shape)            #(10886, 8)

y = train_csv['count']
print(y, y.shape)            # (10886,)



x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.7328, test_size=0.2672)


# 출력값이 '- (음수)'가 나온다면 '활성화 함수'라는 relu라는 함수를 사용한다. 즉, 양수는 어떤수가 나오던 다 쓰되, 음수가 나온다면 0으로 고정한다.


model = Sequential()
model.add(Dense(10, activation='relu', input_dim=8))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')           
model.fit(x_train, y_train, epochs=50, batch_size=12, validation_split=0.8125)



print("========================================")
y_predict = model.predict(x_test, batch_size = 12)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)


mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)

def RMSE(y_test, y_predict):  #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


# exit()
######################## submission.csv 만들기// count 컬럼에 값을 넣어준다. ############################
print(submission)
y_submit = model.predict(test_csv)

submission['count'] = y_submit
print(submission)
print(submission.shape)


# submission.to_csv(path + 'submit/' + 'submit_0904_1720.csv')


