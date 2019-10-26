import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import GraphicUtils as gu


data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

colorDivideConst = 255.0

print('размер тестового массива test_images был : {0} '.format(test_images.shape))
train_images = train_images / colorDivideConst
test_images = test_images / colorDivideConst

print('размер тестового массива test_images стал : {0} '.format( test_images.shape))


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=3)

prediction = model.predict(test_images)


# На данном этапе мне захотелось чтобы предсказание осуществлялось на рандомной картнке из тестового пака.
# Берем одну картинку из проверочного dataset случайным образом для предсказания класса.
randomImageIndex = np.random.randint(10001)
print('Рандомный номер картинки из массива test_images:')
print(randomImageIndex)

randomImage = test_images[randomImageIndex]
np.set_printoptions(threshold=np.inf, precision=2)
print(randomImage)
print('Размер тестовой картинки [x] из массива test_images:')
print(randomImage.shape)

# Модели tf.keras оптимизированы для предсказаний на пакетах (batch) данных, или на множестве примеров сразу.
# Добавляем изображение в пакет данных, состоящий только из одного элемента.
randomImage = (np.expand_dims(randomImage, 0))
print('Размер тестового массива одной картинки: {0}', randomImage.shape)


prediction_single = model.predict(randomImage)

print('Массив распределения предсказания:')
print(prediction_single)

plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
gu.plot_image(randomImageIndex, prediction, test_labels, test_images)
plt.subplot(1, 2, 2)
gu.plot_value_array(randomImageIndex, prediction, test_labels)
plt.xticks(range(10), gu.class_names, rotation=45)
plt.show()

print('Имя предсказанного класса: {0}'.format(gu.class_names[np.argmax(prediction_single)]))

