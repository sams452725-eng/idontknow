import numpy as np
from keras.preprocessing.image import ImageDataGenerator      

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D, Input
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split



#1. 데이터

train_datagen = ImageDataGenerator(
    rescale=1./255,            
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path = './_data/image/men_women/'

xy_data = train_datagen.flow_from_directory(
    path,     
    target_size=(100,100),           # 임의로 넣을수있지만 원데이터 형태로 하기 위해서 150으로 한다.
    batch_size=27167,
    class_mode='binary',        
    color_mode='rgb',   
    shuffle=True,
)

x = xy_data[0][0]
y = xy_data[0][1]


x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2, 
    shuffle=True,      
    random_state=42,
)

print(x_train.shape, y_train.shape)        #(25, 150, 150, 3) (25,)
print(x_test.shape, y_test.shape)          #(7, 150, 150, 3) (7,)


# exit()/
np_path = './_data/men_women_npy/'
np.save(np_path + 'men_women_x_train.npy', arr=x_train)    
np.save(np_path + 'men_women_y_train.npy', arr=y_train)     
np.save(np_path + 'men_women_x_test.npy', arr=x_test)       
np.save(np_path + 'men_women_y_test.npy', arr=y_test)  