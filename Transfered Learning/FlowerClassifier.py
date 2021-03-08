import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

import keras 
from keras.applications import MobileNet, VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model


class FlowerClassifier():
    def __init__(self):
        self.initParameter()
    
    def initParameter(self):
        self.batchSize = 32
        self.epochs = 20
        self.imgRows = 224
        self.imgCols = 224
        self.trainPath = '17_flowers/train'
        self.validPath = '17_flowers/validation'
        self.numClass = 17

    def loadPreModel(self, imgRows, imgCols):
        vgg16Model = VGG16(include_top=False, 
        weights= 'imagenet', input_shape=(imgRows, imgCols,3))
        ### Freeze layers
        for layer in vgg16Model.layers:
            layer.trainable = False
        return vgg16Model

    def buildTopLayers(self, predModel, numClass):
        toplayer = predModel.output
        toplayer = Flatten()(toplayer)
        toplayer = Dense(256, activation = 'relu')(toplayer)
        toplayer = Dropout(0.3)(toplayer)
        toplayer = Dense(numClass, activation = 'softmax')(toplayer)
        return toplayer

    def buildCompleteModel(self, imgRows, imgCols, numClass):
        preModel = self.loadPreModel(imgRows, imgCols)
        topLayer = self.buildTopLayers(preModel, numClass)
        model = Model(inputs=preModel.input, outputs = topLayer)
        print(model.summary)
        return model
    
    def loadDataGeneration(self, trainPath, validPath, imgRows, imgCols, batchSize):
        ### Data Generation
        trainGen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        validGen = ImageDataGenerator(rescale=1./255)

        train_generator = trainGen.flow_from_directory(
            trainPath,
            target_size = (imgRows, imgCols),
            batch_size = batchSize,
            class_mode = 'categorical')

        valid_generator = trainGen.flow_from_directory(
            validPath,
            target_size = (imgRows, imgCols),
            batch_size = batchSize,
            class_mode = 'categorical',
            shuffle = False)
        return train_generator, valid_generator
    
    def callbacksMethod(self):
        checkpoint = ModelCheckpoint('flowerVGG16.h5',
                                    monitor = 'val_loss',
                                    mode = 'min',
                                    save_best_only=True,
                                    verbose=1)
        earlystop = EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=3,
                                verbose=1,
                                restore_best_weights=True)
        callbacks = [earlystop, checkpoint]
        return callbacks
    
    def trainModel(self):
        model = self.buildCompleteModel(self.imgRows, self.imgCols, self.numClass)
        train_generator, valid_generator = self.loadDataGeneration(self.trainPath, self.validPath, self.imgRows, self.imgCols, self.batchSize)
        callbacks = self.callbacksMethod()

        numTrainSamples = 1190
        numValidSamples = 170
        model.compile(loss = 'categorical_crossentropy', optimizer = 'RMSprop', metrics = ['accuracy'])
        history = model.fit_generator(train_generator, 
                                    steps_per_epoch=numTrainSamples//self.batchSize,
                                    epochs = self.epochs, 
                                    callbacks=callbacks, 
                                    validation_data= valid_generator,
                                    validation_steps= numValidSamples//self.batchSize)

model = FlowerClassifier()
model.trainModel()