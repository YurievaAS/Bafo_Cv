import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential 
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(22)

data = pd.read_csv('sign_mnist_train/sign_mnist_train.csv')
data_test = pd.read_csv('sign_mnist_test/sign_mnist_test.csv')

X, y = data.iloc[ : , 1 :] / 255, data.iloc[ : , 0]

X_test, y_test = data_test.iloc[ : , 1 :] / 255, data_test.iloc[ : , 0]

print(y.max())
model_alpha = Sequential([
    Input(shape=(784,),name='Input_layer'),
    Dense(128,activation='relu',name='Hidden_layer_1'),
    Dense(128, activation='relu', name='Hidden_layer_2'),
    Dense(128, activation='relu', name='Hidden_layer_3'),
    Dense(25,activation='softmax',name='Output_layer')
])
model_alpha.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)
model_alpha.fit(
    X,
    to_categorical(y),
    epochs=7,
    batch_size=32
)
model_alpha.evaluate(X_test,to_categorical(y_test))

model_alpha.save('model_alpha.h5')

