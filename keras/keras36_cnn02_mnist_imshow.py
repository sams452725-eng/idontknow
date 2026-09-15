import numpy as np
from tensorflow.keras.datasets import mnist     #CNN을 하는 사람들은 항상 mnist를 쓴다.
import pandas as pd

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)
print(x_train[0])
# print(x_train[0][0])

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) ------------> 4차원 데이터 표기를 할라면 숫자 하나하나 []처리를 해야하기 때문에 생략한거 같다. 또한 흑백은 마지막 shape가 무조건 1이기 때문에 보통 생략을 한다.
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64)) -----> 0이 5923개, 1이 6742개 등등

print(pd.value_counts(y_test))

import matplotlib.pyplot as plt
plt.imshow(x_train[0], 'gray')  # 60000개 중에 0번째를 보여달라는 것이다.
plt.show()