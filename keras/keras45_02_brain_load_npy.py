# 45-1 카피
import numpy as np
from keras.preprocessing.image import ImageDataGenerator      

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D, Input
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping




#1. 데이터
'''
train_datagen = ImageDataGenerator(
    rescale=1./255,         
    # horizontal_flip=True,   
    # vertical_flip=True,     
    # width_shift_range=0.1,  
    # height_shift_range=0.1, 
    # rotation_range=5,       
    # zoom_range=1.2,           
    # shear_range=0.7,        
    # fill_mode='nearest'     
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train,     
    target_size=(150,150),           # 임의로 넣을수있지만 원데이터 형태로 하기 위해서 150으로 한다.
    batch_size=160,
    class_mode='binary',        
    color_mode='grayscale',   
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(150,150),
    batch_size=120,
    class_mode='binary',        
    color_mode='grayscale',    
)

x_train = xy_train[0][0] 
y_train = xy_train[0][1] 
x_test = xy_test[0][0] 
y_test = xy_test[0][1] 

# print(x_train.shape, y_train.shape)        #(160, 150, 150, 1) (160,)
# print(x_test.shape, y_test.shape)          #(120, 150, 150, 1) (120,)

np_path = './_data/kaggle_cat_dog_npy/'
# np.save(np_path + 'keras45_01_x_train.nppy', arr=x_train)                                          
# np.save(np_path + 'keras45_01_y_train.nppy', arr=y_train)    
# np.save(np_path + 'keras45_01_x_test.nppy', arr=x_test)       
# np.save(np_path + 'keras45_01_y_test.nppy', arr=y_test)       
'''

np_path = './_data/kaggle_cat_dog_npy/'
x_train = np.load(np_path + 'keras45_01_x_train.npy')
y_train = np.load(np_path + 'keras45_01_y_train.npy')
x_test = np.load(np_path + 'keras45_01_x_test.npy')
y_test = np.load(np_path + 'keras45_01_y_test.npy')

print(x_train.shape, y_train.shape)   #(160, 150, 150, 1) (160,)
print(x_test.shape, y_test.shape)     #(120, 150, 150, 1) (120,)




exit()
#2. 모델구성
model = Sequential() 

input1 = Input(shape=(150, 150, 1))

conv1_1 = Conv2D(64, (5, 5), padding='same', activation='relu')(input1)
conv1_2 = Conv2D(64, (3, 3), activation='relu')(conv1_1)
pool1   = MaxPool2D()(conv1_2)
drop1   = Dropout(0.2)(pool1)

conv2_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(drop1)
conv2_2 = Conv2D(32, (3, 3), activation='relu')(conv2_1)
pool2   = MaxPool2D()(conv2_2)
drop2   = Dropout(0.25)(pool2)

conv3_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(drop2)
conv3_2 = Conv2D(16, (3, 3), activation='relu')(conv3_1)
pool3   = MaxPool2D()(conv3_2)
drop3   = Dropout(0.3)(pool3)

flt    = Flatten()(drop3)
dense1  = Dense(10, activation='relu')(flt)
drop4   = Dropout(0.3)(dense1)
dense2  = Dense(10, activation='relu')(drop4)
drop5   = Dropout(0.2)(dense2)
dense3  = Dense(5, activation='relu')(drop5)
drop6   = Dropout(0.1)(dense3)
output1 = Dense(1, activation='sigmoid')(drop6)
model   = Model(inputs=input1, outputs=output1)

# model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'],
)        
es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=130,
        restore_best_weights=True,
)
start_time = time.time()            
hist = model.fit(x_train, y_train, 
            epochs = 1000,
            batch_size = 13,
            verbose=1,
            callbacks=[es],
            validation_split=0.3,
)
end_time = time.time()  



#4. 평가,예측
print('================= model.evaluate =================')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1], 4))

y_predict = model.predict(x_test)

y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time - start_time, 2), '초')



'''
1차 
oss : 0.6928420066833496
acc : 0.5
ccuracy_score :  0.5
걸린시간 :  9.33 초

2차
loss : 0.05023930221796036
acc : 0.9875
accuracy_score :  0.9875
걸린시간 :  14.98 초

3차
loss : 0.07323253154754639
acc : 0.975
accuracy_score :  0.975
걸린시간 :  14.6 초

4차
loss : 0.6931713223457336
acc : 0.5
accuracy_score :  0.5
걸린시간 :  7.86 초

5차
loss : 0.019187403842806816
acc : 0.9937
accuracy_score :  0.99375
걸린시간 :  15.18 초

6차
loss : 0.2409275472164154
acc : 0.975
accuracy_score :  0.975
걸린시간 :  17.87 초

7차
loss : 0.23474740982055664
acc : 1.0
accuracy_score :  1.0
걸린시간 :  30.55 초

8차
loss : 0.17622606456279755
acc : 1.0
accuracy_score :  1.0
걸린시간 :  97.02 초












'''