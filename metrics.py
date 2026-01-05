import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from processing import x_train, y_train, x_test
from tensorflow import keras

model = keras.load_model("credit_model.h5")

train_probs = model.predict(x_train) # returns probabilities

train_preds = (train_probs > 0.5).astype(int)