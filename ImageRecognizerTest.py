import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ImageRecogn as ir
import Constants as cn
import GraphicUtils as gu


def verfiy_educated_model_on_test_dataset():

    # build model 
    imageRecognizerModel = ir.assembleModel()


    data = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = data.load_data()


    prediction = imageRecognizerModel.predict(test_images)

    # На данном этапе мне захотелось чтобы предсказание осуществлялось на рандомной картнке из тестового пака.
    # Берем одну картинку из проверочного dataset случайным образом для предсказания класса.
    randomImageIndex = np.random.randint(10001)
    print('Рандомный номер картинки из массива test_images:')
    print(randomImageIndex)

    randomImage = test_images[randomImageIndex]
    np.set_printoptions(threshold=np.inf, precision=2)
    print(randomImage)
    print(f'Размер случайной картинки из массива test_images: {randomImage.shape}')

    # Модели tf.keras оптимизированы для предсказаний на пакетах (batch) данных, или на множестве примеров сразу.
    # Добавляем изображение в пакет данных, состоящий только из одного элемента.
    randomImage = (np.expand_dims(randomImage, 0))
    print('Размер массива одной картинки: {0}' .format(randomImage.shape))


    prediction_single = imageRecognizerModel.predict(randomImage)

    print('Массив распределения предсказания: {0}' .format(prediction_single))

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    gu.plot_image(randomImageIndex, prediction, test_labels, test_images, cn.class_names)
    plt.subplot(1, 2, 2)
    gu.plot_value_array(randomImageIndex, prediction, test_labels)
    plt.xticks(range(10), cn.class_names, rotation=45)
    plt.show()

    print('Имя предсказанного класса: {0}'.format(cn.class_names[np.argmax(prediction_single)]))




    #  Далее мне захотелось взять картинку из интернета и попробовать получить верное предсказание на ней.
    #  Картинка приводится к формату тестовых изображений keras.

def loan_random_image_and_recognize():
        
    # build model 
    imageRecognizerModel = ir.assembleModel()


    data = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = data.load_data()


    img = mpimg.imread(f'mtn\\test_images_pack\\{np.random.randint(1,3)}.jpg')
    # np.set_printoptions(threshold=np.inf, precision=1)
    np.set_printoptions(precision=1)
    img_before = img / 255.0
    img_after = img_before[: , : , 0]
    print('Массив изображения до логического умножения и замены {0}' .format(img_before))

    Substituting_value = 0
    img_after[img_after >= 0.98] = Substituting_value

    print('Массив изображения после замены {0}' .format(img_after))

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img_before)
    a.set_title('Before')
    plt.colorbar(orientation='horizontal')
    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(img_after, cmap = 'binary')
    a.set_title('After')
    plt.colorbar(orientation='horizontal')
    plt.show()

    print(img_after.shape)
    plt.imshow(img_after)
    plt.show()

    # Добавляем изображение в пакет данных, состоящий только из одного элемента.
    img_after = (np.expand_dims(img_after,0))

    print('Размер массива одного изображения:')
    print(img_after.shape)

    # Попытаемся получить какое-нибудь рапределение предсказания.
    prediction_web = ir.assembleModel().predict(img_after)

    print('Массив распределения предсказания:{0}' .format(prediction_web))

    img_numbr = 0
    plt.figure(figsize=(10,3))
    plt.subplot(1,2,1)
    a.set_title('Img')
    gu.nonlabled_plot_image(img_numbr, prediction_web, img_after, )
    plt.subplot(1,2,2)
    a.set_title('Distribution')
    gu.plot_value_array(img_numbr, prediction_web, test_labels)
    plt.xticks(range(10), cn.class_names, rotation=45)
    plt.show()

    print('Имя предсказанного класса:{0}'.format(cn.class_names[np.argmax(prediction_web)]))

