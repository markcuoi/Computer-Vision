import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

import keras 
from keras.applications import MobileNet
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

class MonkeyModel():
    def __init__(self):
        self.initParameter()
    
    def initParameter(self):
        self.imgRows = 224
        self.imgCols = 224
        self.numClass = 10
        self.batchSize = 32
        self.trainPath = './monkey_breed/train'
        self.validPath = './monkey_breed/validation'
        self.epochs = 5
        self.batchSize = 32
        self.numTrainSamples = 1097
        self.numValidSamples = 272
        
    def loadDataset(self, trainPath, validPath, imgRows, imgCols, batchSize):
        trainDatagen = ImageDataGenerator(
                rescale = 1./255,
                rotation_range = 45,
                width_shift_range=0.3,
                height_shift_range=0.3,
                horizontal_flip=True,
                fill_mode='nearest')
        
        validationDatagen = ImageDataGenerator(rescale=1./255)

        trainGenerator = trainDatagen.flow_from_directory(
            trainPath,
            target_size = (imgRows, imgCols),
            batch_size = batchSize,
            class_mode='categorical')
        
        validationGenerator = validationDatagen.flow_from_directory(
            validPath,
            target_size = (imgRows, imgCols),
            batch_size = batchSize,
            class_mode='categorical')
    
        return trainGenerator, validationGenerator

    def loadPreModel(self, imgRows, imgCols):
        MobileNetModel = MobileNet(weights = 'imagenet',
                            include_top = False,
                            input_shape = (imgRows, imgCols, 3))
        
        # Layers are set to trainable as True by default
        for layer in MobileNetModel.layers:
            layer.trainable = False
        return MobileNetModel
            
    def viewModel(self):
        for idx, layer in enumerate(MobileNet.layers):
            print(str(i) + " " + layer.__class__.__name__, layer.trainable)
    
    def createTopLayer(self, pretrainLayer, numClass):
        topLayer = pretrainLayer.output
        topLayer = GlobalAveragePooling2D()(topLayer)
        topLayer = Dense(1024, activation = 'relu')(topLayer)
        topLayer = Dense(1024, activation = 'relu')(topLayer)
        topLayer = Dense(512, activation =  'relu')(topLayer)
        topLayer = Dense(numClass, activation = 'softmax')(topLayer)
        return topLayer
    
    def completeModel(self, imgRows, imgCols, numClass):
        MobileNet = self.loadPreModel(imgRows, imgCols)
        topLayer = self.createTopLayer(MobileNet, numClass)
        model = Model(inputs = MobileNet.input, outputs = topLayer)
        print(model.summary()) 
        return model
    
    def modelCallbacks(self):
        checkpoint = ModelCheckpoint('monkey_breed.h5',
                                monitor = 'val_loss',
                                mode = 'min',
                                save_best_only = True,
                                verbose = 1)
        
        earlystop = EarlyStopping(monitor = 'val_loss',
                                min_delta = 0,
                                patience = 3,
                                verbose = 1,
                                restore_best_weights = True)
        ### put callbacks to callback list
        callbacks = [earlystop, checkpoint]
        return callbacks

    def trainModel(self):
        imgRows = self.imgRows
        imgCols = self.imgCols
        numClass = self.numClass
        epochs = self.epochs
        batchSize = self.batchSize
        
        numTrainSamples = self.numTrainSamples
        numValidSamples = self.numValidSamples

        trainGenerator, validationGenerator = self.loadDataset(self.trainPath, self.validPath, imgRows, imgCols, numClass)
        callbacks = self.modelCallbacks()
        model = self.completeModel(imgRows, imgCols, numClass)
        # We use a very small learning rate 
        model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

        history = model.fit_generator(trainGenerator, 
                    steps_per_epoch = numTrainSamples//batchSize,
                    epochs = epochs,
                    callbacks = callbacks,
                    validation_data = validationGenerator,
                    validation_steps = numValidSamples//batchSize)
    
    def predictModel(self):
        classifier = load_model('./monkey_breed.h5')
        self.monkey_breeds_dict = {"[0]": "mantled_howler ", 
                      "[1]": "patas_monkey",
                      "[2]": "bald_uakari",
                      "[3]": "japanese_macaque",
                      "[4]": "pygmy_marmoset ",
                      "[5]": "white_headed_capuchin",
                      "[6]": "silvery_marmoset",
                      "[7]": "common_squirrel_monkey",
                      "[8]": "black_headed_night_monkey",
                      "[9]": "nilgiri_langur"}

        self.monkey_breeds_dict_n = {"n0": "mantled_howler ", 
                            "n1": "patas_monkey",
                            "n2": "bald_uakari",
                            "n3": "japanese_macaque",
                            "n4": "pygmy_marmoset ",
                            "n5": "white_headed_capuchin",
                            "n6": "silvery_marmoset",
                            "n7": "common_squirrel_monkey",
                            "n8": "black_headed_night_monkey",
                            "n9": "nilgiri_langur"}
        
        for i in range(0,10):
            input_im = self.getRandomImage("./monkey_breed/validation/")
            input_original = input_im.copy()
            input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
            
            input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
            input_im = input_im / 255.
            input_im = input_im.reshape(1,224,224,3) 
            
            # Get Prediction
            res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
            
            # Show image with predicted class
            self.draw_test("Prediction", res, input_original) 
            cv2.waitKey(0)

        cv2.destroyAllWindows()
    
    def draw_test(self, name, pred, im):
        monkey = self.monkey_breeds_dict[str(pred)]
        BLACK = [0,0,0]
        expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
        cv2.putText(expanded_image, monkey, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
        cv2.imshow(name, expanded_image)

    def getRandomImage(self, path):
        """function loads a random images from a random folder in our test path """
        folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
        random_directory = np.random.randint(0,len(folders))
        path_class = folders[random_directory]
        print("Class - " + self.monkey_breeds_dict_n[str(path_class)])
        file_path = path + path_class
        file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
        random_file_index = np.random.randint(0,len(file_names))
        image_name = file_names[random_file_index]
        return cv2.imread(file_path+"/"+image_name) 

mymodel = MonkeyModel()
# mymodel.trainModel()
mymodel.predictModel()