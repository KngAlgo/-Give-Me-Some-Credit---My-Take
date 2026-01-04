import numpy as np
import pandas as pd
from processing import x_train, y_train, x_test, buffer

import tensorflow as tf
from keras import Model, layers, Input
from keras.callbacks import ModelCheckpoint

def credit_model():
    input = Input(shape=(x_train.shape[1]))
    x = layers.Dense(128, activation='relu', kernel_regularizer='l2')(input)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Dense(64, activation='relu', kernel_regularizer="l2")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(32, activation='relu', kernel_regularizer='l2')(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(16, activation='relu', kernel_regularizer="l2")(x)
    
    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(input, output)

    model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

    return model

model = credit_model()

checkpoint = ModelCheckpoint(
    "credit_model.h5", 
    save_best_only=True, 
    monitor='val_loss')

history = model.fit(
    x_train, y_train, 
    epochs=20, 
    batch_size=128, 
    validation_split=0.2, 
    callbacks=[checkpoint])