# keras14_kaggle_bike 카피

# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

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



x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.7, test_size=0.3)


# 출력값이 '- (음수)'가 나온다면 '활성화 함수'라는 relu라는 함수를 사용한다. 즉, 양수는 어떤수가 나오던 다 쓰되, 음수가 나온다면 0으로 고정한다.


model = Sequential()
model.add(Dense(10, activation='relu', input_dim=8))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')      
start_time = time.time()           #현재시간을 반환. 즉, 시작 시간     
model.fit(x_train, y_train, epochs=2, batch_size=8)
end_time = time.time()             #현재시간을 반환. 즉, 끝 시간

# 훈련이 시작하는 시점에 시작시간을 걸고, 끝나는 시점에 끝 시간을 건다.
# 시간이 곧 돈이다. 거래처나 프로젝트에서 납기나 기한이 정해져 있는데 최초 epoch를 돌리고 걸린 시간을 가지고 남은 시간 동안 얼마나 더 돌릴수있는지를 예측하고 유추할수있다.





print("========================================")
y_predict = model.predict(x_test, batch_size = 8)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)


mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)

def RMSE(y_test, y_predict):  #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

print('걸린시간 :', round(end_time - start_time, 2), '초')














'''
######################## submission.csv 만들기// count 컬럼에 값을 넣어준다. ############################
print(submission)
y_submit = model.predict(test_csv)

submission['count'] = y_submit
print(submission)
print(submission.shape)


# submission.to_csv(path + 'submit/' + 'submit_0904_1720.csv')
'''

