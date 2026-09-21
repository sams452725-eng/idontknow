import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

#2. 모델구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1),     #(9,9,10)
                 padding='same',
                 strides=1,
))
model.add(Conv2D(filters=9, kernel_size=(3,3),         #(7,7,9)
                 padding='valid',     
                 strides=1,          
))
#padding = 'valid'가 디폴트 valid or same 두개 중에 하나만 들어간다.
#stride는 보폭이란 뜻으로 커널사이즈로 잘라서 conv를 할때 (2,2)에서 stride=1이라는건 옆으로 1칸 움직이는거라 겹치는거고 stride=2는 2칸이 움직이기 때문에 겹치지 않는다. 겹치지 않게 훈련을 하는건 완전히 비추다.
#batch 사이즈는 숫자가 어떻게 되든 행으로 자르기 때문에 끝에 부족하면 부족한대로 훈련을 돌리지만, stride는 커널사이즈의 shape 형태로 훈련을 돌리기 때문에 끝에 모자른걸 그대로 돌리면 shape이 망가지기 때문에 error가 뜬다. 그래서 부족한 마지막은 버려버린다.
model.summary()