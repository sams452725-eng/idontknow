import numpy as np
from keras.preprocessing.image import ImageDataGenerator      # ImageDataGenerator는 이미지 수치화 + 증폭
print(np.__version__)

train_datagen = ImageDataGenerator(
    rescale=1./255,         # 1.은 부동소수점 형변환 1/255. 해도 된다. 어디든 .을 붙이면 된다.
    horizontal_flip=True,   # 수평(좌우) 뒤집기
    vertical_flip=True,     # 수직(상하) 뒤집기    true는 한다, false는 안한다.
    width_shift_range=0.1,  # 평형이동
    height_shift_range=0.1, 
    rotation_range=5,       # 각도조절(정해진 각도만큼 이미지 회전)
    zoom_range=1.2,           
    shear_range=0.7,        # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동(한마디로 찌부)
    fill_mode='nearest'     # 데이터가 이동 및 증폭하면 데이터가 이동한 쪽은 소실되고 그전에 있던 곳은 비워지게 된다. 비워진 곳은 새롭게 채워야 하는데 그때 근처에 있는 수치로 채운다.
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
)
# test_datagen은 절대 변환하면 안된다. rescale은 수치화지만 나머지는 시험지 자체를 조작하는거기 때문에 절대 하면 안된다.

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train,      #경로     직접 입력해도 되지만 귀찮으니까 위에 path를 따로 잡았다.
    target_size=(100,100),
    batch_size=10,
    class_mode='binary',        # 이진분류 -----> 우리가 찾는건 nomal or ad 이기 때문이다.
    color_mode='grayscale',     # 흑백
    shuffle=True,
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size=10,
    class_mode='binary',        
    color_mode='grayscale',    
    # shuffle=True,    # test에서는 필요가 없다.
)
# Found 120 images belonging to 2 classes.

# print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x0000017CBA987F70>
# print(xy_train.next())       # Iterator의 첫번째를 보여줘
# print(xy_train.next())       # 두번째 Iterator를 출력해줘

# print(xy_train[0])
# print(xy_train[1])
# print(xy_train[2])

# print(xy_train[0][0])       # 첫번째 배치의 x데이터
# print(xy_train[0][1])       # 첫번째 배치의 y데이터

print(xy_train[0][0].shape)          # (10, 100, 100, 1)
print(xy_train[0][1].shape)          # (10,)

# print(xy_train[15][0])             # if [16][0] = ValueError: Asked to retrieve element 16, but the Sequence has length 16 이유는 총 160장에 batch_size = 10이기 때문이다.

print(type(xy_train))           # <class 'keras.preprocessing.image.DirectoryIterator'>     Directory = folder
print(type(xy_train[0]))        # <class 'tuple'>   list와 비슷하지만 수정이 불가하다.
print(type(xy_train[0][0]))     # <class 'numpy.ndarray'>
print(type(xy_train[0][1]))     # <class 'numpy.ndarray'>
