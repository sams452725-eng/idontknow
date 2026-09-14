# validation 자체는 원칙적으로 훈련에 영향을 미치지 않는다.
# epoch를 한번만 돌리는게 아니라 굉장히 많이 돌리게 되면 사람이 val-loss를 보고 test 값을 판단하기 때문에 간접적으로 영향을 미치게 된다.
# 하지만 원칙적으로는 training(훈련)에 직접적으로 영향을 미치는건 아니다.
# train validation test 순서로 가는데 훈련 쪽지시험 모의고사의 개념으로 생각하면 된다.



########## keras15 복사 ##########


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array(range(1, 11))
y = np.array(range(1, 11))

x_train = np.array(range(1,8))
y_train = np.array(range(1,8))

x_test = np.array(range(8,11))
y_test = np.array(range(8,11))

x_val = np.array([7,8])
y_val = np.array([7,8])

x_test = np.array([9,10])
y_test = np.array([9,10])



print('x.shape :', x.shape)
print('y.shape :', y.shape)

# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 1))

# 3. 컴파일, 훈련
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

# 하이퍼파라미터: 에폭, 레이어의 깊이, 노드 개수, 배치 사이즈

