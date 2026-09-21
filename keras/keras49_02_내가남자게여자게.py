import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import time
from keras.preprocessing.image import ImageDataGenerator      
from sklearn.metrics import accuracy_score




#1. 데이터
np_path = './_data/men_women_npy/'
x_train = np.load(np_path + 'men_women_x_train.npy')
y_train = np.load(np_path + 'men_women_y_train.npy')
x_test = np.load(np_path + 'men_women_x_test.npy')
y_test = np.load(np_path + 'men_women_y_test.npy')



#2. 모델
model_path = './_save/men_women/mw_0921_1657_0011-0.1944.keras'
model = load_model(model_path)

#4. 평가,예측
predict_path = './_data/img_to_array_npy/'
x = np.load(predict_path + 'keras48_me.npy')
x = x/255.0



print('================= model.evaluate =================')
y_predict = model.predict(x)

score = y_predict[0][0]

print('================= 판별 결과 =================')
print(f"예측 수치값 (0=여자, 1=남자): {score:.4f}")
if score < 0.5:
    woman_percent = (1 - score) * 100
    print(f"축하합니다! 당신은 {woman_percent:.2f}% 확률로 [여자] 입니다!")
else:
    man_percent = score * 100
    print(f"축하합니다! 당신은 {man_percent:.2f}% 확률로 [남자] 입니다!")


# 예측 수치값 (0=여자, 1=남자): 0.9301
# 축하합니다! 당신은 93.01% 확률로 [남자] 입니다!