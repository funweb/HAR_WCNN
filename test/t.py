import numpy as np
import keras
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.layers import Input, GlobalAveragePooling2D
from keras import models
from keras.models import Model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import tensorflow

nClasses = 8
X, y = make_classification(n_samples=100000, n_features = 2304, n_informative = 200, n_classes = nClasses)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

n_values = np.max(y) + 1
y_train = np.eye(n_values)[y_train]
y_test = np.eye(n_values)[y_test]

shape_x = 48
shape_y = 48


X_train = X_train.reshape(75000,shape_x,shape_y,1)
X_test = X_test.reshape(25000,shape_x,shape_y,1)

input_img = Input(shape=(shape_x, shape_y, 1))

### 1st layer
layer_1 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
layer_1 = Conv2D(10, (3,3), padding='same', activation='relu')(layer_1)

### 2nd layer
layer_2 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
layer_2 = Conv2D(10, (5,5), padding='same', activation='relu')(layer_2)

### 3rd layer
layer_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
layer_3 = Conv2D(10, (1,1), padding='same', activation='relu')(layer_3)

### Concatenate
mid_1 = keras.layers.concatenate([layer_1, layer_2, layer_3], axis = 3)

flat_1 = Flatten()(mid_1)

dense_1 = Dense(1200, activation='relu')(flat_1)
dense_2 = Dense(600, activation='relu')(dense_1)
dense_3 = Dense(150, activation='relu')(dense_2)

output = Dense(nClasses, activation='softmax')(dense_3)


model = Model([input_img], output)

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 512
epochs = 1

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

