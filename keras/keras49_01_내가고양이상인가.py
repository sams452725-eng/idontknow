'''
개고양이 가중치를 가져와서 모델 완성
데이터는 개 고양이 npy데이터 사용
내 사진도 npy 불러와서 predict만 하면 되겠지요

끝!!!!
'''
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
np_path = './_data/kaggle_cat_dog_npy/'
x_train = np.load(np_path + 'catdog_x_train.npy')
y_train = np.load(np_path + 'catdog_y_train.npy')
x_test = np.load(np_path + 'catdog_x_test.npy')
y_test = np.load(np_path + 'catdog_y_test.npy')



#2. 모델
model_path = './_save/kaggle_CatDog/CatDog_0918_1754_0088-0.3324.keras'
model = load_model(model_path)

#4. 평가,예측
predict_path = './_data/img_to_array_npy/'
x = np.load(predict_path + 'keras48_me.npy')
x = x/255.0



print('================= model.evaluate =================')
y_predict = model.predict(x)

score = y_predict[0][0]

print('================= 판별 결과 =================')
print(f"예측 수치값 (0=고양이, 1=개): {score:.4f}")
if score < 0.5:
    cat_percent = (1 - score) * 100
    print(f"🐱 축하합니다! 당신은 {cat_percent:.2f}% 확률로 [고양이상] 입니다!")
else:
    dog_percent = score * 100
    print(f"🐶 축하합니다! 당신은 {dog_percent:.2f}% 확률로 [개상] 입니다!")