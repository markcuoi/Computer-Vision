import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

class NeuralNetwork():
    def __init__(self, batch_size = 32, epochs = 20):
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataProc()

    def loadData(self):
        (self.x_train_orig, self.y_train_orig),(self.x_test_orig, self.y_test_orig) = cifar10.load_data()
    def dataProc(self):
        self.loadData()
        (x_train, y_train), (x_test, y_test) = (self.x_train_orig, self.y_train_orig),(self.x_test_orig, self.y_test_orig)
        img_rows = x_train[0].shape[0]
        img_cols = x_train[0].shape[1]
        self.input_shape = (x_train.shape[0], img_rows, img_cols, 3)
        x_train = x_train.astype('float32')
        x_test =  x_test.astype('float32')

        x_train/=255.0
        x_test/=255.0

        self.num_classes = 10

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
    def buidAlexNet(self):
        model = Sequential()
        # 1st Conv Layer 
        model.add(Conv2D(96, (11, 11), input_shape=self.x_train.shape[1:],
            padding='same', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 2nd Conv Layer 
        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 3rd Conv Layer 
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 4th Conv Layer 
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(1024, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # 5th Conv Layer 
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(1024, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 1st FC Layer
        model.add(Flatten())
        model.add(Dense(3072))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # 2nd FC Layer
        model.add(Dense(4096))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # 3rd FC Layer
        model.add(Dense(self.num_classes))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

        print(model.summary())
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
        return model
    def trainModel(self):
        model = self.buidAlexNet()
        history = model.fit(self.x_train, self.y_train,
                            batch_size = self.batch_size,
                            epochs = self.epochs,
                            validation_data =(self.x_test, self.y_test),
                            shuffle = True)
a = NeuralNetwork()
a.trainModel()
 