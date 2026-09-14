import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras. layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping


############### acc >=.95 ###############


#1. 데이터
datasets = load_wine()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets['target']
# print(x.shape, y.shape)        #(178, 13) (178,)
# print(y)        #(178,)
# print(np.unique(y, return_counts=True))       #(array([0, 1, 2]), array([59, 71, 48]))
# print(pd.Categorical(y))


from tensorflow.keras.utils import to_categorical
y=to_categorical(y)
# print(y,y.shape)        # (178, 3)


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=.8,
    shuffle=True,
    random_state=338,
    stratify=y,
)

# print(x_train.shape, x_test.shape)          # (142, 13) (36, 13)
# print(y_train.shape, y_test.shape)          # (142, 3) (36, 3)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))   


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'],)     
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=50,
    restore_best_weights=True,
)          
start_time = time.time()
model.fit(x_train, y_train, epochs=1500, batch_size=100,
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
# print(y_predict)
y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)  
y_test = np.argmax(y_test, axis=1)
# print(y_test)     

# exit()
accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score :', accuracy_score)
# print('걸린시간 :', round(end_time - start_time, 2), '초')


# loss : 0.12591630220413208
# acc : 0.97
# acc_score : 0.9722222222222222