import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import GraphicUtils as gu



#  Build the main model


def assembleModel():
        
    data = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = data.load_data()

    colorDivideConst = 255.0

    train_images = train_images / colorDivideConst
    test_images = test_images / colorDivideConst

    print('размер тестового массива test_images: {0} '.format(test_images.shape))

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Компиляция модели
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=1)

    return model






