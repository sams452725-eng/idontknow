from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D     # Convolution은 원 이미지를 커널 사이즈로 쪼개는거다.

model = Sequential()   # 행에 들어가는 데이터 수는 얼마든 들어갈수있기 때문에 Sequential에서는 통상 None으로 반출이된다.
model.add(Conv2D(10, (2,2), input_shape=(5,5,1)))      # Dense의 input_dim의 형식과 다른건 필터의 단위이기 때문이다.  2차원에서 데이터의 개수가 행으로 와서 ( , ) 앞에 써지는 행이 데이터의 개수다. 같은 개념으로 4차원에서 ( , , , )의 맨 앞에 오는 숫자도 행의 개념으로 데이터의 개수다.
model.add(Conv2D(5, (2,2)))        #  Output Shape (None, 3, 3, 5) --------> 쉽게 계산하는 방법 (input 크기 - layer 크기 + 1). ex) 5-2+1 = 4, 4-2+1 = 3

model.summary()
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 4, 4, 10)          50        
                                                                 
#  conv2d_1 (Conv2D)           (None, 3, 3, 5)           205       
                                                                 
# =================================================================
# Total params: 255
# Trainable params: 255
# Non-trainable params: 0

