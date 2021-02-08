import os
from os import listdir
from os.path import join, isfile
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model

mypath = './datasets/images/'
class createDataset():
    def __init__(self, dog_dir_train = "./datasets/catsvsdogs/train/dogs/", 
                dog_dir_val = "./datasets/catsvsdogs/validation/dogs/",
                cat_dir_train = "./datasets/catsvsdogs/train/cats/",
                cat_dir_val = "./datasets/catsvsdogs/validation/cats/"):

        self.dog_dir_train = dog_dir_train
        self.dog_dir_val = dog_dir_val
        self.cat_dir_train = cat_dir_train
        self.cat_dir_val = cat_dir_train

        self.dog_count = 0
        self.cat_count = 0
        self.size = 256
        self.training_size = 1000
        self.test_size = 500
        self.training_labels = []
        self.test_labels = []
        self.training_images = []
        self.test_images = []
  
        self.loadDogCatImage(self.getFilenames(mypath))
        self.saveNPZ()
    def getFilenames(self, mypath):
        filenames = [i for i in listdir(mypath) if isfile(join(mypath,i))]
        return filenames

    def makeDir(self):
        print('hehehehehhe')
        self.sub_makeDir(self.dog_dir_train)
        self.sub_makeDir(self.dog_dir_val)
        self.sub_makeDir(self.cat_dir_train)
        self.sub_makeDir(self.cat_dir_val)

    def sub_makeDir(self, dataDir):
        if os.path.exists(dataDir):
            shutil.rmtree(dataDir)
        os.makedirs(dataDir)

    def loadDogCatImage(self, filenames):
        self.makeDir()
        for i, filename in enumerate(filenames):
            print('check1',filename)
            if filename[0] == 'd':
                self.dog_count += 1
                image = cv2.imread(mypath+filename)
                image = cv2.resize(image, (self.size, self.size), interpolation = cv2.INTER_AREA)
                if self.dog_count <= self.training_size:
                    self.training_images.append(image)
                    self.training_labels.append(1)
                    cv2.imwrite(self.dog_dir_train + 'dog' + str(self.dog_count) + '.jpg', image)
                if self.dog_count > self.training_size and self.dog_count <= self.training_size + self.test_size:
                    self.test_images.append(image)
                    self.test_labels.append(1)
                    cv2.imwrite(self.dog_dir_val + 'dog' + str(self.dog_count-self.training_size) + '.jpg', image)

            if filename[0] == 'c':
                self.cat_count += 1
                image = cv2.imread(mypath+filename)
                image = cv2.resize(image, (self.size, self.size), interpolation = cv2.INTER_AREA)
                if self.cat_count <= self.training_size:
                    self.training_images.append(image)
                    self.training_labels.append(0)
                    cv2.imwrite(self.cat_dir_train + 'cat' + str(self.cat_count) +'.jpg', image)
                if self.cat_count >self.training_size and self.cat_count <= self.training_size + self.test_size:
                    self.test_images.append(image)
                    self.test_labels.append(0)
                    cv2.imwrite(self.cat_dir_val + 'cat' + str(self.cat_count-1000)+'.jpg',image)
            if self.dog_count == self.training_size + self.test_size and self.cat_count == self.training_size + self.test_size:
                break
        print("Training and Test Data Extraction Complete") 
          
    def saveNPZ(self):
        # Using numpy's savez function to store our loaded data as NPZ files
        np.savez('./dataNPZ/cats_vs_dogs_training_data.npz', np.array(self.training_images))
        np.savez('./dataNPZ/cats_vs_dogs_training_labels.npz', np.array(self.training_labels))
        np.savez('./dataNPZ/cats_vs_dogs_test_data.npz', np.array(self.test_images))
        np.savez('./dataNPZ/cats_vs_dogs_test_labels.npz', np.array(self.test_labels))           

class modelTraining():
    def __init__(self, batch_size = 16, epochs = 1):
        
        # Model config
        self.batch_size = batch_size
        self.epochs = epochs

        # data argumentation path
        self.train_data_dir = './datasets/catsvsdogs/train'
        self.validation_data_dir = './datasets/catsvsdogs/validation'
        self.nb_train_samples = 2000
        self.nb_validation_samples = 1000
        self.img_width = 256
        self.img_height = 256
        self.dirNPZ = './dataNPZ/cats_vs_dogs'
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.loadDataset(self.dirNPZ)

    def loadDataset(self, dirNPZ):
        npzfile = np.load(dirNPZ + '_training_data.npz')
        train = npzfile['arr_0']
        npzfile = np.load(dirNPZ + '_training_labels.npz')
        train_labels = npzfile['arr_0']
        npzfile = np.load(dirNPZ + '_test_data.npz')
        test = npzfile['arr_0']
        npzfile = np.load(dirNPZ + '_test_labels.npz') 
        test_labels = npzfile['arr_0']
        return (train, train_labels), (test, test_labels)

    def visualizeImage(self, numberImg):
        for i in range(numberImg):
            random = np.random.randint(0, len(self.x_train))
            cv2.imshow('image_'+str(i), self.x_train[random])
            print('{} - cat'.format(i)) if self.y_train[random] == 0  else print('{} - dog'.format(i))
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def dataPreProcess(self):
        y_train = self.y_train.reshape(self.y_train.shape[0],1)
        y_test = self.y_test.reshape(self.y_test.shape[0],1) 

        x_train = self.x_train.astype('float32')
        x_test = self.x_test.astype('float32')

        x_train /= 255
        x_test /= 255
        return (x_train, y_train), (x_test, y_test)

    def buildModel(self):
        (x_train, x_test), (y_train, y_test) = self.dataPreProcess()    
        img_rows = x_train[0].shape[0]
        img_cols = x_train[0].shape[1]
        self.input_shape = (img_rows, img_cols, 3)

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = self.input_shape))
        self.model.add(Conv2D(64, (3, 3), activation = 'relu'))        
        self.model.add(MaxPooling2D((4,4)))

        self.model.add(Conv2D(128, (3,3 ), activation = 'relu'))
        self.model.add(MaxPooling2D((4,4)))
        self.model.add(Conv2D(512, (3,3 ), activation = 'relu'))
        self.model.add(Dropout(0.25))
        self.model.add(MaxPooling2D((4,4)))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation = 'relu'))
        self.model.add(Dense(64, activation = 'relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(1, activation= 'sigmoid'))

        self.model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics =['acc'])
        print(self.model.summary())
        return self.model

    def trainModel(self):
        (x_train, y_train), (x_test, y_test) = self.dataPreProcess()    
        self.model = self.buildModel()
        self.history = self.model.fit(x_train, y_train, batch_size = self.batch_size, 
                                        epochs = self.epochs, validation_data = (x_test, y_test), shuffle = True)
        self.model.save("./modelWeight/cats_vs_dogs_V1.h5")
        self.scores = self.model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', self.scores[0])
        print('Test accuracy:', self.scores[1])
        self.plotGraph()

    def Generator(self):
        validation_datagen = ImageDataGenerator(rescale = 1./255)
        #creating our data generator
        train_datagen = ImageDataGenerator(
            rescale = 1./255,
            rotation_range = 30,
            width_shift_range = 0.3,
            height_shift_range = 0.3,
            horizontal_flip = True,
            fill_mode = 'nearest')

        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_height, self.img_width,),
            batch_size = self.batch_size,
            class_mode = 'binary',
            shuffle = True)

        validation_generator = validation_datagen.flow_from_directory(
            self.validation_data_dir,
            target_size = (self.img_height, self.img_width),
            batch_size = self.batch_size,
            class_mode = 'binary', 
            shuffle = False) 
        return train_generator, validation_generator

    def trainModelWithGen(self):
        (x_train, y_train), (x_test, y_test) = self.dataPreProcess()    
        train_generator, validation_generator = self.Generator()
        self.model = self.buildModel()
        self.history = self.model.fit_generator(
                train_generator,
                steps_per_epoch = self.nb_train_samples // self.batch_size,
                epochs = self.epochs,
                validation_data = validation_generator,
                validation_steps = self.nb_validation_samples // self.batch_size)
        
        self.model.save("./modelWeight/cats_vs_dogs_Gen_V1.h5")
        self.scores = self.model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', self.scores[0])
        print('Test accuracy:', self.scores[1])
        self.plotGraph()
        # model.fit_generator(dataAugmentaion.flow(trainX, trainY, batch_size = 32),
        #  validation_data = (testX, testY), steps_per_epoch = len(trainX) // 32,
        #  epochs = 10)
    
    def plotGraph(self):
        history_dict = self.history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)
        line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')
        line2 = plt.plot(epochs, loss_values, label='Training Loss')
        plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
        plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
        plt.xlabel('Epochs') 
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.show()

        history_dict = self.history.history
        acc_values = history_dict['acc']
        val_acc_values = history_dict['val_acc']
        epochs = range(1, len(loss_values) + 1)          
        line1 = plt.plot(epochs, val_acc_values, label='Validation/Test Accuracy')
        line2 = plt.plot(epochs, acc_values, label='Training Accuracy')
        plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
        plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
        plt.xlabel('Epochs') 
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.show()

    def drawPredict(self, name, pred, input_im):
        BLACK = [0,0,0]
        pred='cat' if pred == "[0]" else 'dog'
        expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, self.imageL.shape[0] ,cv2.BORDER_CONSTANT,value=BLACK)
        #expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
        cv2.putText(expanded_image, str(pred), (700, 250) , cv2.FONT_HERSHEY_COMPLEX_SMALL,4, (0,255,0), 2)
        cv2.imshow(name, expanded_image)

    def modelPredict(self, modelPath = 'modelWeight/cats_vs_dogs_V1.h5'):
        (x_train, y_train), (x_test, y_test) = self.dataPreProcess()    
        classifier = load_model(modelPath)
        for i in range(len(x_test)):
            rand = np.random.randint(0, len(x_test))
            input_im = x_test[rand]

            self.imageL = cv2.resize(input_im, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
            input_im = input_im.reshape(1,256,256,3)

            #get Prediction
            pred = str(classifier.predict_classes(input_im, 1, verbose = 0)[0])
            self.drawPredict('Prediction', pred, self.imageL) 
            cv2.waitKey(0)
        cv2.destroyAllWindows()


data = createDataset()
model = modelTraining()
model.visualizeImage(10)
model.trainModel()
model.trainModelWithGen()
model.modelPredict()

