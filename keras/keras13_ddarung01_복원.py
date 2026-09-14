# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
from tensorflow. keras. models import Sequential
from tensorflow. keras. layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터


# 원래는 path = "./_data/ddarung/파일명.csv" 이렇게 해야하지만, 반복되는 부분을 줄이기 위해 path를 지정해준다.
# 반복되는 부분이란 : 아래에서 각각의 pd.read_csv(path + "파일명.csv")에서 파일 경로는 동일하지만 파일명이 다르기 때문에 path를 지정해주면 반복되는 부분을 줄일 수 있다.


path = "./_data/ddarung/"                #상대경로
# path = "c:/study/_data/ddarung/"         #절대경로
# path = "c:\study\_data\ddarung\"         #\s를 인식해서 에러
# path = "c://study//_data//ddarung//"
# path = "c:\\study\\_data\\ddarung\\"
# path = "c:\\study\\_data\\ddarung/"      #섞어쓰기는 되지만 가독성이 떨어지기 때문에 가급적 섞어쓰기는 피하는 것이 좋다.




train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)         #[1459 rows x 10 columns]
# id열 포함 [1459 rows x 11 columns]
# id열 제거 [1459 rows x 10 columns] index_col=0 때문


test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)

# [715 rows x 9 columns]


submission = pd.read_csv(path + 'submission.csv', index_col=0)
print(submission)

# [715 rows x 1 columns]
# NAN은 결칙치 즉, 결측치가 존재한다는 의미이다. 결측치를 처리해야 한다.
# 즉, index는 데이터가 아니다.


print(train_csv.shape)         #(1459, 10)
print(test_csv.shape)          #(715, 9)                # y값이 없기 때문에 evaluation을 할 수 없다. 즉, test_csv는 x값만 존재한다.
print(submission.shape)        #(715, 1)                # 따라서 test_csv와 submission은 train에 사용될수가 없다. 왜냐하면 y값이 없기 때문이다.



print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='str')


print(train_csv.info())
print(test_csv.info())

# exit()
################################### 결측치 처리 1. 삭제 ########################################
# dropna는 결측치가 존재하는 행을 제거한다. 즉, 결측치가 존재하는 행을 삭제한다.
train_csv = train_csv.dropna()
print(train_csv)            #[1328 rows x 10 columns]

################################### train_csv를 x와 y로 분리 ########################################

x = train_csv.drop(['count'], axis=1)
# axis=1은 열을 의미한다. 즉, 'count'라는 열(컬럼)을 제거한다. 원데이터 에서 'count'라는 열을 제거하고 나머지 열들을 x로 만든다.
# drop은 제거하고 싶은 컬럼명을 입력하면 된다. axis=0은 행을 의미한다. 즉, 행을 제거하고 싶으면 axis=0을 입력하면 된다.

print(x)                   #[1328 rows x 9 columns]

y = train_csv['count']      # 'count'라는 열(컬럼)을 y로 만든다.
print(y)                    #[1328 rows x 1 columns]
print(y.shape)              #(1328,)


############################ 변형참고 ################################

test_csv = pd.read_csv(path + "test.csv", index_col= 0) #model.predict()에 넣을 값
print(test_csv) #[715 * 9] 
print(test_csv.shape)
print(test_csv.columns)
print(test_csv.info())

# (열 컬럼 특성 속성 피쳐 어트리뷰트) 전부 똑같은 말이다. 

submission = pd.read_csv(path+"submission.csv" , index_col= 0) #여기에 model.predict(submission) 값을 넣는다.
print(submission) # NaN은 결측치를 의미, [715 * 9] 
print(submission.shape)
print(submission.columns)

x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, test_size=0.2)

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=9))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)

#4. 평가, 예측

print("========================================")
y_predict = model.predict(x_test, batch_size = 32)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)

def RMSE(y_test, y_predict):  #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


'''
하이퍼파라미터 튜닝
#1. 데이터
random_state
train_size

#1 모델
layer의 깊이
node의 갯수

#2. 컴파일, 훈련
epoch
batch_size
'''

'''
1차 시도
random : 337
train_size : 0.75
epoch = 50
batch_size = 1

결과
rmse : 37
r2 : 0.66
'''


'''
2차 시도
random : 337
train_size : 0.75
epoch = 200
batch_size = 32

결과
rmse : 27
r2 : 0.67              #반드시 각 회차 시도별로 돌리는 상황 명세서를 만들어놔야 한다.
'''