import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import GraphicUtils as gu
import random
import constants as c
import model as m


m.imgRecognModel.fit(m.train_images, m.train_labels, epochs=20)
test_loss, test_acc = m.imgRecognModel.evaluate(m.test_images,  m.test_labels, verbose=2)

print('\nТочность на проверочных данных:', test_acc)

# На данном этапе мне захотелось чтобы предсказание осуществлялось на рандомной картнке из тестового пака.
# Берем одну картинку из проверочного dataset случайным образом для предсказания класса.
randomImageIndex = np.random.randint(10001)
print('\nРандомный номер картинки из массива m.test_images:',randomImageIndex)

# Модели tf.keras оптимизированы для предсказаний на пакетах (batch) данных, или на множестве примеров сразу.
# Поэтому, создаем пакет данных состоящий только из одного элемента.
randomImage = m.test_images[randomImageIndex]
randomImage = (np.expand_dims(randomImage, 0))
print('\nРазмер массива одной картинки:', randomImage.shape)

randomLabel = m.test_labels[randomImageIndex]
randomLabel = (np.expand_dims(randomLabel, 0))
print('\nРазмер массива m.test_images:', randomLabel.shape)

np.set_printoptions(precision=2)
prediction = m.imgRecognModel.predict(randomImage)
print('\nМассив предсказания prediction:', prediction)

fig = plt.figure(figsize=(10,3))
a = plt.subplot(1, 2, 1)
a.set_title('Image')
gu.plot_image(0, prediction, randomLabel, randomImage)
a = plt.subplot(1, 2, 2)
a.set_title('Prediction')
gu.plot_value_array(0, prediction, randomLabel)
plt.xticks(range(10), c.class_names, rotation=45)
plt.show()

print('\nИмя предсказанного класса:', c.class_names[np.argmax(prediction)])

# Далее мне захотелось взять картинку из интернета и попробовать получить верное предсказание на ней.
# Картинка приводится к формату тестовых изображений keras.
np.set_printoptions(threshold=np.inf, precision=2)
First_image = mpimg.imread(f'mtn\\test_images_pack\\{0}.jpg')

First_image = First_image / c.pixScaleDivConst
First_image = First_image[: , : , 0]
Substituting_value = 0
First_image[First_image >= 0.97] = Substituting_value

print('\nПриведение шкалы цветности к единице, логическое умножение и замена 1 на 0:', First_image)

Webpack_images = np.array([First_image])
print('\nРазмер до добавления изображений:', Webpack_images.shape)

# Алгоритм обработки и добавления изображений в массив
for x in range (1,12):
    for i in range (1,12):
        images = mpimg.imread(f'mtn\\test_images_pack\\{i}.jpg')
        images = images / c.pixScaleDivConst
        images = images[: , : , 0]
        Substituting_value = 0
        images[images >= 0.97] = Substituting_value
        # images = np.array([images]) 
        if i == x:
            break 
    Webpack_images = np.append(Webpack_images, [images], axis = 0)
    print('\nИзменение размера с добавлением картинок в массив:', Webpack_images.shape)


plt.figure(figsize=(10,10))
for i in range(0,12):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(Webpack_images[i], cmap=plt.cm.binary)
plt.show()

# Попытаемся получить какое-нибудь рапределение предсказания.
prediction_web = m.imgRecognModel.predict(Webpack_images)
print('\nМассив распределения предсказания:', prediction_web)

np.set_printoptions(precision=2)
num_rows = 4
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  a = plt.subplot(num_rows, 2*num_cols, 2*i+1)
  a.set_title('Image: {}' .format(i+1), fontsize = 8.0)
  gu.plot_image(i, prediction_web, c.Webimgpack_lables, Webpack_images)
  a = plt.subplot(num_rows, 2*num_cols, 2*i+2)
  a.set_title('Prediction: {}' .format(i+1), fontsize = 8.0)
  plt.tight_layout()
  gu.plot_value_array(i, prediction_web, c.Webimgpack_lables)
plt.show()

Webimgpack_lables = (np.expand_dims(c.Webimgpack_lables, 2))
test_loss, test_acc = m.imgRecognModel.evaluate(Webpack_images,  Webimgpack_lables, verbose=2)
print('\nТочность в selfmade наборе данных:', test_acc)

print('')
print('Введите номер картинки для вывода детального отчета:')

imgInput = int(input())
fig = plt.figure(figsize=(10,3))
a = plt.subplot(1, 2, 1)
a.set_title('Image {}' .format(imgInput))
gu.plot_image(imgInput, prediction_web, c.Webimgpack_lables, Webpack_images)
a = plt.subplot(1, 2, 2)
a.set_title('Prediction')
gu.plot_value_array(imgInput, prediction_web, c.Webimgpack_lables)
plt.xticks(range(10), c.class_names, rotation=45)
plt.show()
