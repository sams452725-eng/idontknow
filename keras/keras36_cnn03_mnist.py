# 36-2 카피

import numpy as np
from tensorflow.keras.datasets import mnist     #CNN을 하는 사람들은 항상 mnist를 쓴다.
import pandas as pd


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) 
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(np.max(x_train), np.min(x_train)) # 255 0    ------> 일반적으로 이미지는 최대값이 255이기 때문에 굳이 print를 안해봐도 쉽게 스케일링을 할수있다. 
print(np.max(x_test), np.min(x_test))   # 255 0

###### 스케일링 1 ######
# x_train = x_train/255. # 뒤에 '.' 을 붙이는 이유는 산출된 값을 파이썬의 float 형식으로 바꿔주기 위함이다.(파이썬 기초)
# x_test = x_test/255. 
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test))   # 1.0 0.0

###### 스케일링 2 ######
x_train = (x_train-127.5)/127.5 
x_test = (x_test-127.5)/127.5
print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
print(np.max(x_test), np.min(x_test))   # 1.0 -1.0
