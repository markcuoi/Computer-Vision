import keras
from keras.preprocessing import image
from keras.applications import vgg16, inception_v3, resnet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from keras.preprocessing import image

class PretrainedModel():
    def __init__(self, batch_size = 16, epochs = 20):
        self.batch_size = batch_size 
        self.epochs = epochs
        self.loadData()
        self.size = 224
    def loadData(self, mypath = './images'):
        images = ['./images/'+i for i in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, i))]
        return images

    def loadModel(self):
        vgg_model = vgg16.VGG16(weights = 'imagenet')
        inception_model = inception_v3.InceptionV3(weights = 'imagenet')
        resnet_model = resnet50.ResNet50(weights = 'imagenet')
        return vgg_model, inception_model, resnet_model

    def compareModel(self):
        image_lst = []
        images = self.loadData()
        vgg_model, inception_model, resnet_model= self.loadModel()
        for i in images:
            image =cv2.imread(i)
            image = cv2.resize(image, (self.size, self.size), interpolation = cv2.INTER_AREA)
            image = image.reshape((1,)+image.shape)
            image_lst.append(image)
        
        image_lst_np = np.array([img for img in image_lst])
        print(image_lst_np)
        for idx,img in enumerate(image_lst_np):

            #load image using opencv
            img2 = cv2.imread(images[idx])
            imageL = cv2.resize(img2, None, fx=.5, fy=.5, interpolation = cv2.INTER_CUBIC) 
            # Get VGG16 Predictions
            preds_vgg_model = vgg_model.predict(img)
            preditions_vgg = decode_predictions(preds_vgg_model, top=3)[0]
            self.draw_test("VGG16 Predictions", preditions_vgg, imageL) 
            
            # # # Get Inception_V3 Predictions
            # preds_inception = inception_model.predict(img)
            # preditions_inception = decode_predictions(preds_inception, top=3)[0]
            # self.draw_test("Inception_V3 Predictions", preditions_inception, imageL) 

            # Get ResNet50 Predictions
            preds_resnet = resnet_model.predict(img)
            preditions_resnet = decode_predictions(preds_resnet, top=3)[0]
            self.draw_test("ResNet50 Predictions", preditions_resnet, imageL) 
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_test(self, name, preditions, input_im):
        """Function displays the output of the prediction alongside the orignal image"""
        BLACK = [0,0,0]
        expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, input_im.shape[1]+300 ,cv2.BORDER_CONSTANT,value=BLACK)
        img_width = input_im.shape[1]
        for (i,predition) in enumerate(preditions):
            string = str(predition[1]) + " " + str(predition[2])
            cv2.putText(expanded_image,str(name),(img_width + 50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),1)
            cv2.putText(expanded_image,string,(img_width + 50,50+((i+1)*50)),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),1)
        cv2.imshow(name, expanded_image)

a = PretrainedModel()     
a.compareModel()

