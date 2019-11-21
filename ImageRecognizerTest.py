import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import GraphicUtils as gu
import random
import constants as c
import model as m


"""Here model is compiled"""
m.imgRecognModel.fit(m.train_images, m.train_labels, epochs=1)
test_loss, test_acc = m.imgRecognModel.evaluate(m.test_images,  m.test_labels, verbose=2)

print('\nТочность на проверочных данных:', '%.2f' % (test_acc*100), '%')

"""Random picking from 0 to 10000 to use lable prediction"""
randomImageIndex = np.random.randint(10001)
print('\nРандомный номер картинки из массива m.test_images:',randomImageIndex)

def random_image():
    """Модели tf.keras оптимизированы для предсказаний на пакетах (batch) данных, 
    или на множестве примеров сразу. Поэтому, необходимо создать пакет данных 
    состоящий только из одного элемента."""
    randomImage = m.test_images[randomImageIndex]
    randomImage = (np.expand_dims(randomImage, 0))
    return(randomImage)
print('\nРазмер массива одной картинки:', random_image())

def random_lable():
    """A single object array must be done for random lable 
    related to the random"""
    randomLabel = m.test_labels[randomImageIndex]
    randomLabel = (np.expand_dims(randomLabel, 0))
    return(randomLabel)
print('\nРазмер массива одного лейбла:', random_lable())

def one_img_prediction():
    """prediction of the only one image"""
    np.set_printoptions(precision=2)
    prediction = m.imgRecognModel.predict(random_image())
    return(prediction)
print('\nМассив предсказания prediction:', one_img_prediction(), end='\n')
print('\nИмя предсказанного класса:', c.class_names[np.argmax(one_img_prediction())], end='\n')

def verfiy_trained_model_on_test_image():
    """Plotting random image and predicted distribution""" 
    fig = plt.figure(figsize=(10,3))
    a = plt.subplot(1, 2, 1)
    a.set_title('Image')
    gu.plot_image(0, one_img_prediction(), random_lable(), random_image())
    a = plt.subplot(1, 2, 2)
    a.set_title('Prediction')
    gu.plot_value_array(0, one_img_prediction(), random_lable())
    plt.xticks(range(10), c.class_names, rotation=45)
plt.show(verfiy_trained_model_on_test_image())


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
        if i == x:
            break 
    Webpack_images = np.append(Webpack_images, [images], axis = 0)
    print('\nИзменение размера с добавлением картинок в массив:\n', Webpack_images.shape, end='')


plt.figure(figsize=(10,10))
for i in range(0,12):
    a=plt.subplot(5,5,i+1)
    a.set_title('Image: {}' .format(i), fontsize = 8.0)
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
  a.set_title('Image: {}' .format(i), fontsize = 8.0)
  gu.plot_image(i, prediction_web, c.Webimgpack_lables, Webpack_images)
  a = plt.subplot(num_rows, 2*num_cols, 2*i+2)
  a.set_title('Prediction: {}' .format(i), fontsize = 8.0)
  plt.tight_layout()
  gu.plot_value_array(i, prediction_web, c.Webimgpack_lables)
plt.show()

Webimgpack_lables = (np.expand_dims(c.Webimgpack_lables, 2))
test_loss, test_acc = m.imgRecognModel.evaluate(Webpack_images,  Webimgpack_lables, verbose=2)
print('\nТочность в selfmade наборе данных:', '%.2f' % (test_acc*100), '%', end='\n')

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
