import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


# path = './_data/kaggle_santander/'
path = 'c:/study/_data/kaggle_santander/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# print(train_csv.shape)           #(200000, 201)
# print(test_csv.shape)            #(200000, 200)    -----------> 뒤에 200이 input_dim에 들어가는 숫자다.
# print(submission_csv.shape)      #(200000, 1)

# print(train_csv.info())
# print(train_csv.isna().sum())          #결측치 확인
# print(test_csv.isnull().sum())         #결측치 확인

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
# print(x.shape, y.shape)           # (200000, 200) (200000,)

y = np.array(y)           #sklearn에서 reshape 할때 pandas로 땡겨온 csv같은 파일의 y값은 arrange 형태가 나온다. 그걸 numpy형식으로 바꿔줄때 쓰는 식이다.
# y = y.to_numpy()


# print(np.unique(y, return_counts=True))           #(array([0, 1]), array([179902,  20098]))

#################### ONEHOT 3. sklearn ####################
from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(150,1)      #(150, 1)
y = y.reshape(-1,1)      #(150, 1)        ---------------------> 늘상 y값의 범위를 알수없기 때문에 앞에 -1을 쓰면 1부터 전체 y의 범위의 제일 끝까지 범위가 설정된다.
# print(y, y.shape)

# exit()
# ohe = OneHotEncoder()       # sparse라는 혼돈행렬의 형태로 값이 반환된다.
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y)
# print(y)
# reshape는 절대 1.내용(값)과 2.순서를 바꾸면 안된다. 바뀐걸 알았으면 '데이터 조작', 몰랐으면 '데이터 오염'이다.




# y = pd.get_dummies(y, dtype=int)
# # print(y, y.shape)       #(200000, 2)



# exit()
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=.8,
    shuffle=True,
    random_state=333,
    stratify=y,
)


model = Sequential()
model.add(Dense(512, input_dim=200, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))   


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'],)     
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=100,
    restore_best_weights=True,
)          
start_time = time.time()
model.fit(x_train, y_train, epochs=5000, batch_size=256,
          verbose=2,
          validation_split=.2,
          callbacks=[es],
          )
end_time = time.time()


#4. 평가,예측
result = model.evaluate(x_test, y_test,)
print('loss :', result[0])
print('acc :', round(result[1], 4))

y_predict = model.predict(x_test)
# print(y_predict)
y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)  
y_test = np.argmax(y_test, axis=1)
# print(y_test)     

# exit()
accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score :', accuracy_score)
print('걸린시간 :', round(end_time - start_time, 2), '초')


y_submit = model.predict(test_csv)

submission_csv['target'] = y_submit[:, 1]
# print(submission_csv)
# print(submission_csv.shape)



submission_csv.to_csv(path + 'submit/' + 'submit_0910_1431.csv')
