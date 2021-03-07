import pickle 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.utils import np_utils
from keras.models import load_model

class confustionMatrix():
    def __init__(self, batch_size = 128, epochs = 20):
        self.batch_size = batch_size
        self.epochs = epochs
        self.loadData()
        self.procData()
    def loadData(self):
        (self.x_trainOrig,self.y_trainOrig), (self.x_testOrig,self.y_testOrig) = mnist.load_data()

    def procData(self):
        (x_train, y_train), (x_test, y_test) = (self.x_trainOrig,self.y_trainOrig), (self.x_testOrig,self.y_testOrig)
        img_rows= x_train[0].shape[0]
        img_cols = x_train[0].shape[1]
        self.input_shape = (img_rows, img_cols, 1)

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        x_train.astype('float32')
        x_test.astype('float32')

        self.x_train = x_train/255
        self.x_test = x_test/255

        self.y_train = np_utils.to_categorical(y_train)
        self.y_test = np_utils.to_categorical(y_test)
        self.num_classes = self.y_train.shape[1]

    def builModel(self):
        model = Sequential()
        model.add(Conv2D(32,(3, 3), activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation = 'relu'))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation = 'softmax'))
        
        model.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=['acc'])
        print(model.summary())
        return model
    
    def trainModel(self):
        model = self.builModel()
        self.history = model.fit(self.x_train, self.y_train,
                                batch_size=self.batch_size,
                                epochs=self.epochs,
                                verbose=1,
                                validation_data=(self.x_test, self.y_test))
        self.scores = model.evaluate(self.x_test, self.y_test, verbose=0)
        print('test loss: ', self.scores[0])
        print('test accuracy: ', self.scores[1])

        # save our model and history file
        model.save('./ModelWeight/mnist.h5')

        pickle_out = open("MNIST_history.pickle","wb")
        pickle.dump(self.history.history, pickle_out)
        pickle_out.close()
        self.plotGraph()

    def loadHisFile(self):
        pickle_in = open("MNIST_history.pickle","rb")
        saved_history = pickle.load(pickle_in)
        print(saved_history)
    
    def plotGraph(self):
        history_dict = self.history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values)+1)

        line1 = plt.plot(epochs , val_loss_values, label = 'Validation/Test Loss')
        line2 = plt.plot(epochs, loss_values, label = 'Training Loss')
        plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
        plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

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

    # def plotConfusionMatrix(self):
    #     modelPath=''
    #     classifier = model_load(modelPath)
    #     y_pred  = classifier.predict_classes(self.x_test)
    #     print(classification_report(np.argmax(y_test,axis=1), y_pred))
    #     print(confusion_matrix(np.argmax(y_test,axis=1), y_pred)) = confustionMatrix()

a = confustionMatrix()
a.trainModel()