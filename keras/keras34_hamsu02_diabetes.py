# keras33-2 카피


from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout, Input
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time





#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target



x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.75, 
    # test_size=0.25, 
    # shuffle=True,       # 디폴트 섞는다.
    random_state=4333,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler           #preprocessing 이건 전처리다. OneHot에서 써봤다.
from sklearn.preprocessing import RobustScaler
# scaler = MinMaxScaler()                                    #내가 받은 데이터가 무조건 쓰레기라고 상정을 해야된다. 내가 받은 데이터가 전처리가 안된 데이터가 대부분일꺼기 때문이다.
# scaler = StandardScaler()
# scaler = MaxAbsScaler()                                      
scaler = RobustScaler()               
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


print(np.min(x_train), np.max(x_train))      # 0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test))        # -0.0012367054167697258 1.0


#2. 모델구성
# model = Sequential()
# model.add(Dense(10, input_dim=10))
# model.add(Dropout(.2))
# model.add(Dense(10))
# model.add(Dropout(.3))
# model.add(Dense(10))
# model.add(Dropout(.5))
# model.add(Dense(10))
# model.add(Dense(1))



input1 = Input(shape=(10,))     
dense1 = Dense(10,)(input1)      
drop1 = Dropout(.2)(dense1)
dense2 = Dense(10,)(drop1)         
drop2 = Dropout(.3)(dense2)
dense3 = Dense(10,)(drop2)         
drop3 = Dropout(.5)(dense3)
dense4 = Dense(10,)(drop3)         
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=30,
                   restore_best_weights=True,
                   verbose=1,
                   )



# exit()

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



#4. 평가, 예측
print("========================================")
loss = model.evaluate(x_test, y_test,)# batch_size = 18)
print("========================================")
y_predict = model.predict(x_test,)# batch_size = 18)

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
r2:  0.45289994197336125
mse:  3089.723888241847
RMSE :  55.58528481749326

dropout
r2:  0.37922657423227424
mse:  3505.7910424986276
RMSE :  59.20972084462675

함수형
r2:  0.3780563698531164
mse:  3512.399721060005
RMSE :  59.265501947254315


'''