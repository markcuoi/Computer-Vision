import cv2
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

class dataAugmentaion():
    def __init__(self, image_path = 'dog.jpeg'):
        self.image_path = image_path
        self.datagen()
        print('hehe')
    def datagen(self):
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        img = cv2.imread(self.image_path)
        inputGen = np.array(img)
        inputGen = inputGen.reshape((1,)+inputGen.shape)
        i = 0
        for batch in datagen.flow(inputGen, save_to_dir='output', save_prefix='dog', save_format='jpeg'):
            i += 1
            print('hehe')
            if i > 35:
                break 

a = dataAugmentaion()
# a.datagen
