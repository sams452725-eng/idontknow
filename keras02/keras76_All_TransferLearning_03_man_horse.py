import numpy as np
from keras.preprocessing.image import ImageDataGenerator      

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D, Input
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import time
from keras.preprocessing.image import ImageDataGenerator      
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet201, DenseNet169, DenseNet121
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetMobile, NASNetLarge
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications import Xception
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
import random
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#1. 데이터
np_path = './_save/horse-human_npy/'
x_train = np.load(np_path + 'keras46_01_x_train.npy')
y_train = np.load(np_path + 'keras46_01_y_train.npy')
x_test = np.load(np_path + 'keras46_01_x_test.npy')
y_test = np.load(np_path + 'keras46_01_y_test.npy')



start_time=time.time()


random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)


model_list = [VGG19(include_top=False, input_shape=(32,32,3)),
              ResNet50(include_top=False, input_shape=(32,32,3)),
              ResNet101(include_top=False, input_shape=(32,32,3)), 
              DenseNet201(include_top=False, input_shape=(32,32,3)),
            #   InceptionV3(),
            #   InceptionResNetV2(),
              MobileNetV2(include_top=False, input_shape=(32,32,3)),
              EfficientNetB0(include_top=False, input_shape=(32,32,3)),]




for model_i in model_list:
    model_i.trainable = False

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(10, activation='softmax'))
    

    #3. 컴파일, 훈련
    from tensorflow.keras.optimizers import Adam
    learning_rate = 0.005

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=14,
        restore_best_weights=True,
)
    rlr = ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        patience=20,
        verbose=1,
        factor=0.5,        
)  
    model.fit(x_train, y_train, epochs=1000, batch_size=32,
              verbose=1,
              validation_split=.2,
              callbacks=[es,rlr])

end_time = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0)
print('='*60)
print('모델명 :', model_i.name, 'loss :', loss[0], 'acc :', loss[1])
print('걸린시간 : ', round(end_time - start_time, 2), '초')


