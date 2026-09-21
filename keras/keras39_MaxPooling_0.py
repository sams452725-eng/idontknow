import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D   #통상적으로 MaxPooling2D는 Conv를 먼저하고 그 뒤에 쓴다.



#2. 모델구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1),     #(10,10,10)
                 padding='same',
                 strides=1,
))
model.add(MaxPooling2D())
model.add(Conv2D(filters=9, kernel_size=(3,3),         #(8,8,9)
                 padding='valid',     
                 strides=2,          
))
model.summary()