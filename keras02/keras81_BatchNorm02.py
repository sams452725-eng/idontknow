# 36-2 카피

import time

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Sequential

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)


# 2. 모델구성
model = Sequential()
model.add(Conv2D(64, (5, 5), input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(4, 4), activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(GlobalAveragePooling2D())
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dropout(0.2))
model.add(Dense(units=16, activation="relu"))
model.add(Dense(10, activation="softmax"))

# 3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

start_time = time.time()
model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=128,
    verbose=1,
    validation_split=0.2,
)
end_time = time.time()

# 4. 평가,예측
print("================= model.evaluate =================")
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss :", loss[0])
print("acc :", loss[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)
y_test = np.argmax(y_test, axis=1).reshape(-1, 1)

acc_score = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc_score)
print("걸린시간 : ", round(end_time - start_time, 2), "초")


# 0.995 맞추기

# None, None, batchnorm, relu
# loss : 0.4158577620983124
# acc : 0.8658000230789185
# 313/313 [==============================] - 0s 1ms/step
# accuracy_score :  0.8658
# 걸린시간 :  102.38 초

# relu, relu, batchnorm, relu
# loss : 0.053567077964544296
# acc : 0.9836999773979187
# 313/313 [==============================] - 0s 1ms/step
# accuracy_score :  0.9837
# 걸린시간 :  105.68 초
