from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt



path = 'c:/study/_data/image/'
# load_img는 1장짜리 사진을 불러오기에 좋다.
img = load_img(path +'handsomeguy.jpg', target_size=(150,150))

print(img)
# <PIL.Image.Image image mode=RGB size=150x150 at 0x17B60B89630>
print(type(img))
# <class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
# print(arr)
print(arr.shape)      #(150, 150, 3)
print(type(arr))      #<class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis=0)   # 0번째에 차원 증가------------> reshape를 해도 되지만, 이런 방법도 있다.
# print(arr)
print(arr.shape)      #(1, 150, 150, 3)

# np_path = './_data/img_to_array_npy/'
# np.save(np_path + 'keras48_me.npy', arr=arr)     

###################### 여기부터 증폭 ######################

datagen = ImageDataGenerator(
    rescale=1./255,         # 1.은 부동소수점 형변환 1/255. 해도 된다. 어디든 .을 붙이면 된다.
    # horizontal_flip=True,   # 수평(좌우) 뒤집기
    # vertical_flip=True,     # 수직(상하) 뒤집기    true는 한다, false는 안한다.
    width_shift_range=0.1,  # 평형이동
    # height_shift_range=0.1, 
    rotation_range=15,       # 각도조절(정해진 각도만큼 이미지 회전)
    zoom_range=0.5,           
    # shear_range=0.7,        # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동(한마디로 찌부)
    fill_mode='nearest'     # 데이터가 이동 및 증폭하면 데이터가 이동한 쪽은 소실되고 그전에 있던 곳은 비워지게 된다. 비워진 곳은 새롭게 채워야 하는데 그때 근처에 있는 수치로 채운다.
)

# it = Iterator 그냥 정의
it = datagen.flow(arr, 
                  batch_size=1)
print(it)
# <keras.preprocessing.image.NumpyArrayIterator object at 0x000001E34D867F70>

# print(it.next())      # 파이썬 3.10까지
# print(next(it))       # 파이썬 3.10이후

# 53번째 줄 = 54번째 줄. 형식만 다르지 같은 의미다.

print(next(it).shape)        #(1, 150, 150, 3)

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))
for i in range(5):            #0~4까지
    batch = next(it)
    # print(batch.shape)
    batch = batch.reshape(150,150,3)     # reshape는 순서와 값이 바뀌면 절대 안되기 때문에 (1, 150, 150, 3)--->(150,150,3) 가능하지만, 다른거로는 형변형이 안된다.

    ax[i].imshow(batch)
    ax[i].axis('off')
plt.show()
