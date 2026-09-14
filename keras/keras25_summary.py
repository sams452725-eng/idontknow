from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))      # input_dim = input_layer 즉, y의 node값이 1이다.
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))                   # 마지막 dense = output_layer

model.summary()
