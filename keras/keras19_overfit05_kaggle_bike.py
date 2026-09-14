import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error
import time

#1. 데이터
path = './_data/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)

print(train_csv)  #  [10886 rows x 11 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)

print(test_csv)  #  [6493 rows x 8 columns]

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)

print(submission)  #  [6493 rows x 1 columns]

print(train_csv.shape)
print(test_csv.shape)
print(submission.shape)



print(train_csv.info())
print(test_csv.info())
print(submission.info())

print(train_csv.describe())
############################### 결측치 확인 ###############################
print(train_csv.isna().sum())
print(train_csv.isnull().sum())


############################### x,y 분리 ###############################
x = train_csv.drop(['casual','registered','count'], axis = 1)
print(x)
y = train_csv['count']
print(y, y.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.75, 
    # test_size=0.25, 
    # shuffle=True,       # 디폴트 섞는다.
    random_state=4333,
)

x_val, x_test, y_val, y_test = train_test_split(
    x, y, 
    train_size=0.75, 
    # test_size=0.25, 
    # shuffle=True,       # 디폴트 섞는다.
    random_state=4333,
)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()           #현재시간을 반환. 즉, 시작 시간     
hist = model.fit(x_train, y_train, epochs = 500, batch_size = 18,
            verbose=2,
            validation_data = (x_val, y_val),
            )
end_time = time.time()             #현재시간을 반환. 즉, 끝 시간


#4. 평가, 예측

print("========================================")
loss = model.evaluate(x_test, y_test, batch_size = 18)
print("========================================")
y_predict = model.predict(x_test, batch_size = 18)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)

def RMSE(y_test, y_predict):  #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# r2:  0.269730806350708
# mse:  24519.009765625
# RMSE :  156.58547111921015

# print('걸린시간 :', round(end_time - start_time, 2), '초')





# print("================= history =======================")
# print(hist)
# print("================= hist.history =======================")
# print(hist.history)
# print("================= loss =======================")
# print(hist.history['loss'])
# print("================= val_loss =======================")
# print(hist.history['val_loss'])
# print("========================================")



# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss')          # y값만 넣으면 시간순으로 그려줌.
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')        #[18:]을 넣는건 그래프를 시각화 했을 때 특정값이 튀면 나머지가 뒤에서 y=0쪽에서 수럽하니까 애초에 튀는 그래프를 빼버리는거다. 
# plt.legend(loc='upper right')           # 우측 상단에 라벨표시
# plt.title('Kaggle_Bike LOSS')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()           # 격자표시 추가
# plt.show()