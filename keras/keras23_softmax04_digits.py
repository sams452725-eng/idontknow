import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential
from tensorflow.keras. layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

# acc = 1.0


#1. 데이터
datasets = load_digits()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

# exit()
x = datasets.data
y = datasets['target']
# print(x.shape, y.shape)        #(581012, 54) (581012,)
# print(y)        #(581012,)
# print(np.unique(y, return_counts=True))       #(array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))
# print(pd.Categorical(y))





# exit()
from tensorflow.keras.utils import to_categorical
y=to_categorical(y)
# print(y,y.shape)        # (581012, 8)

# exit()
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=.8,
    shuffle=True,
    random_state=338,
    stratify=y,
)

# print(x_train.shape, x_test.shape)          # (464809, 54) (116203, 54)
# print(y_train.shape, y_test.shape)          # (464809, 8) (116203, 8)

# exit()
#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=64, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))   


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'],)     
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=15,
    restore_best_weights=True,
)          
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=72,
          verbose=1,
          validation_split=.2,
          callbacks=[es],
          )
end_time = time.time()


#4. 평가,예측
result = model.evaluate(x_test, y_test,)
print('loss :', result[0])
print('acc :', round(result[1], 2))

y_predict = model.predict(x_test)
print(y_predict)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)  
y_test = np.argmax(y_test, axis=1)
print(y_test)     

# exit()
accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score :', accuracy_score)
print('걸린시간 :', round(end_time - start_time, 2), '초')

# loss : 0.3211411237716675
# acc : 0.92
# acc_score : 0.9222222222222223
# 걸린시간 : 11.21 초