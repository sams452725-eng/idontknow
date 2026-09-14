from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split

# 1 . 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))


# x_train = np.array(range(1,9))
x_train = x[:8]
# y_train = np.array(range(1,9))
y_train = y[:8]

# x_val = np.array([9,13])
x_val = x[8:12]
# y_val = np.array([9,13])
y_val = y[8:12]

# x_test = np.array([13,17])
x_test = x[12:]
# y_test = np.array([13,17])
y_test = y[12:]
# [실습]train_test_split으로 잘라라

train_test_split(x,y, test_size=.75, random_state=111)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.25, random_state=111)
x_val, x_test, y_val, y_test = train_test_split(x_test,y_test, test_size=.25, random_state=111)
print(x_train)          
print(x_val)            
print(x_test)           


model = Sequential()
model.add(Dense(1, input_dim = 1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=4,
            verbose=1,
            validation_data = (x_val, y_val),
            )
# verbose = 0 : 침묵
# verbose = 1 : 디폴트
# verbose = 2 : 프로그래스바 삭제
# verbose = 3 : Epoch만 나옴
# verbose = 4 : Epoc만 나옴
# verbose = 5 : 침묵
# verbose = 나머지 : Epoc만 나옴

# verbose는 성능에 영향을 미치지 않는다. 다만 보여지기만 간략하게 보여진다.





# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
print('loss : %.8f' % loss)

result = model.predict(np.array([[10]]))
print('predict :', result)
