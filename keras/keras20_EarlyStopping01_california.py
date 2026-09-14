from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import time


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# 틀린 코드
# x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, test_size=0.3,random_state=333)
# x_val, x_test, y_val, y_test = train_test_split(x,y,train_size=0.5, test_size=0.5,random_state=333)

# 올바른 코드
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=333)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5, test_size=0.5, random_state=333)




#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(10, activation='relu'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))        #output Dense에 아무것도 activation하지 않으면 디폴트 리니어(일반 선형)다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=50,
        restore_best_weights=True,
)

start_time = time.time()           #현재시간을 반환. 즉, 시작 시간     
hist = model.fit(x_train, y_train, 
            epochs = 500000,
            batch_size = 32,
            validation_split=.2,
            callbacks=[es],
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

print('걸린시간 :', round(end_time - start_time, 2), '초')





print("================= history =======================")
print(hist)
print("================= hist.history =======================")
print(hist.history)
print("================= loss =======================")
print(hist.history['loss'])
print("================= val_loss =======================")
print(hist.history['val_loss'])
print("========================================")
print(es)



import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')          # y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')        #[18:]을 넣는건 그래프를 시각화 했을 때 특정값이 대략 15~18번째에서 튀면 나머지는 뒤에서 y=0쪽에 안정적으로 수럽하니까 애초에 튀는 그래프를 빼버리는거다. 
plt.legend(loc='upper right')           # 우측 상단에 라벨표시
plt.title('캘리포니아 LOSS')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()           # 격자표시 추가
plt.show()