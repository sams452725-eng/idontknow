# 50-1 카피
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
###################### 여기부터 증폭 ######################

datagen = ImageDataGenerator(
    rescale=1./255,         # 1.은 부동소수점 형변환 1/255. 해도 된다. 어디든 .을 붙이면 된다.
    horizontal_flip=True,   # 수평(좌우) 뒤집기
    # vertical_flip=True,     # 수직(상하) 뒤집기    true는 한다, false는 안한다.
    width_shift_range=0.1,  # 평형이동
    # height_shift_range=0.1, 
    rotation_range=15,       # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=0.5,           
    # shear_range=0.7,        # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동(한마디로 찌부)
    fill_mode='nearest'     # 데이터가 이동 및 증폭하면 데이터가 이동한 쪽은 소실되고 그전에 있던 곳은 비워지게 된다. 비워진 곳은 새롭게 채워야 하는데 그때 근처에 있는 수치로 채운다.
)

augment_size=100
print(x_train.shape)           #(60000, 28, 28)
print(x_train[0].shape)        #(28, 28)

# aaa = np.tile(x_train[0], augment_size)
# print(aaa.shape)               #(28, 2800)

aaa = np.tile(x_train[0], augment_size).reshape(-1,28,28,1)
print(aaa.shape)               #(100, 28, 28, 1)
##### 단순 복붙!!! #####

xy_data = datagen.flow(
        np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),
        np.zeros(augment_size),
        batch_size=augment_size,
        shuffle=False,
).next()

print(xy_data)
print(type(xy_data))

# print(xy_data.data.shape)   #AttributeError: 'tuple' object has no attribute 'data'
print(len(xy_data))  #2 ------> 왜냐하면 x,y 두개니까

print(xy_data[0].shape)   #(100,28,28,1)
print(xy_data[1].shape)   #(100, )

plt.figure(figsize=(7,7))
for i in range(49) :
    plt.subplot(7,7, i+1)
    plt.imshow(xy_data[0][i], cmap='gray')
plt.show()
