import tensorflow as tf
from tensorflow import keras
import constants as c

#Building the main model
data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images / c.pixScaleDivConst
test_images = test_images / c.pixScaleDivConst

imgRecognModel = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Компиляция модели
imgRecognModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
