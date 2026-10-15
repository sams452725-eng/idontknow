# keras75 카피

import warnings
warnings.filterwarnings('ignore')

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

model_list = [VGG19(include_top=False, input_shape=(32,32,3)),
              ResNet50(include_top=False, input_shape=(32,32,3)),
              ResNet101(include_top=False, input_shape=(32,32,3)), 
              DenseNet201(include_top=False, input_shape=(32,32,3)),
            #   InceptionV3(),
            #   InceptionResNetV2(),
              MobileNetV2(include_top=False, input_shape=(32,32,3)),
              EfficientNetB0(include_top=False, input_shape=(32,32,3)),]

import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
import random
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

start_time=time.time()

for model_i in model_list:
    model_i.trainable = False

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # 스케일링
    x_train = x_train/255.
    x_test = x_test/255.

    #3. 컴파일, 훈련
    from tensorflow.keras.optimizers import Adam
    learning_rate = 0.001

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

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('='*60)
    print('모델명 :', model_i.name, 'loss :', loss[0], 'acc :', loss[1])

end_time = time.time()
print('='*60)
print('걸린시간 : ', round(end_time - start_time, 2), '초')

'''
============================================================
모델명 : vgg19 loss : 1.1750131845474243 acc : 0.5878000259399414
============================================================
모델명 : resnet50 loss : 1.5597045421600342 acc : 0.4449000060558319
============================================================
모델명 : resnet101 loss : 1.7221808433532715 acc : 0.38370001316070557
============================================================
모델명 : densenet201 loss : 1.0822938680648804 acc : 0.6283000111579895
============================================================
모델명 : mobilenetv2_1.00_224 loss : 1.794587254524231 acc : 0.349700003862381
============================================================
모델명 : efficientnetb0 loss : 2.3026139736175537 acc : 0.10000000149011612
============================================================
걸린시간 :  2895.95 초















'''