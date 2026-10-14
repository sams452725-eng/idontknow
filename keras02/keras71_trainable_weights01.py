import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

print(tf.__version__)

random.seed(333)           # python의 초기 가중치를 고정한다.
np.random.seed(333)        # np의 초기 가중치를 고정한다. 
tf.random.set_seed(333)    # tf의 초기 가중치를 고정한다.(그럼에도 tf는 틀어지는 경우가 있다.)

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))
# model.summary()
# print("=" * 25, 'model.weights', "=" * 25) 
# print("=" * 60) 
# print(model.weights)
# print("=" * 60) 
# print("=" * 25, 'model.trainable weights', "=" * 25) 
# print("=" * 60) 
# print(model.trainable_weights)
# print("=" * 60) 
# print("=" * 60) 
# print(len(model.weights))             # 6
# print(len(model.trainable_weights))   # 6

################################# 동결 #################################
model.trainable = False      # 동결 ★★★★★
################################# 동결 #################################
print("=" * 60) 
print(len(model.weights))             # 6
print(len(model.trainable_weights))   # 0
print("=" * 25, 'model.trainable weights', "=" * 25) 
print(model.trainable_weights)
print("=" * 25, 'model.weights', "=" * 25) 
print(model.weights)

model.summary()
