import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data = keras.datasets.fashion_mnist
 
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

print('размер тестового массива test_images:')
print(test_images.shape)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
    ])

# Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
 
model.fit(train_images, train_labels, epochs=1)

prediction = model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# На данном этапе мне захотелось чтобы предсказание осуществлялось на рандомной картнке из тестового пака.
# Берем одну картинку из проверочного dataset случайным образом для предсказания класса.
x = np.random.randint(10001)
print('Рандомный номер картинки из массива test_images:')
print(x)

img = test_images[x]
np.set_printoptions(threshold=np.inf, precision=2)
print(img)
print('Размер тестовой картинки [x] из массива test_images:')
print(img.shape)

# Модели tf.keras оптимизированы для предсказаний на пакетах (batch) данных, или на множестве примеров сразу.
# Добавляем изображение в пакет данных, состоящий только из одного элемента.
img = (np.expand_dims(img,0))
print('Размер тестового массива одной картинки:')
print(img.shape)
prediction_single = model.predict(img)

print('Массив распределения предсказания:')
print(prediction_single)

i = x
plt.figure(figsize=(10,3))
plt.subplot(1,2,1)
plot_image(i, prediction, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, prediction, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

print('Имя предсказанного класса:')
print(class_names[np.argmax(prediction_single)])
