# keras74-2 카피
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32,32,3) 
)
vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(10, activation='softmax'))

model.layers[2].trainable=False

import pandas as pd
pd.set_option('max_colwidth', None)
layers = [
    (type(layer).__name__, layer.name, layer.trainable)
    for layer in model.layers
]
results = pd.DataFrame(
    layers,columns=["Type","Name","Trainable"]
)

print(results)