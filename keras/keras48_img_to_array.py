from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt
path = 'c:/study/_data/image/'
# load_img는 1장짜리 사진을 불러오기에 좋다.
img = load_img(path +'handsomeguy.jpg', target_size=(100,100))

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

arr = np.expand_dims(arr, axis=0)   # 0번째에 차원 증가
# print(arr)
print(arr.shape)      #(1, 150, 150, 3)

np_path = './_data/img_to_array_npy/'
np.save(np_path + 'keras48_me.npy', arr=arr)     

