import cv2
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

img = load
# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# create a grid of 3x3 images
for i in range(10)
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i])
plt.show()

class dataAugmentaion():
    def __init__(self, image_path = ''):
    
    def datagen(self):
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        )
        img = cv2.imread('dog.jpeg')
        print(img)
        inputGen = np.array(img)
        print(inputGen)


