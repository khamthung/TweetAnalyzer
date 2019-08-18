import numpy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import initializers

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.misc as misc
from keras.models import model_from_json

from keras.datasets import cifar100
class CIFAR100model(object):

    def __init__(self, num_classed= 100):
        """Return a Customer object whose name is *name* and starting
        balance is *balance*."""
        self.model = Sequential()
        self.num_classed = num_classed
        self.categories = ['apples', 'aquarium fish', 'baby', 'bear', 'beaver',
            'bed', 'bee', 'beetle', 'bicycle', 'bottles',
            'bowls','boy', 'bridge', 'bus', 'butterfly', 
            'camel', 'cans', 'castle', 'caterpillar', 'cattle', 
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 
            'couch', 'crab', 'crocodile', 'cups', 
            'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 
            'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer keyboard', 'lamp', 
            'lawn-mower', 'leopard', 'lion', 'lizard', 'lobster', 
            'man', 'maple', 'motorcycle', 'mountain', 'mouse', 
            'mushrooms', 'oak', 'oranges', 'orchids', 'otter', 
            'palm', 'pears', 'pickup truck', 'pine', 'plain', 
            'plates', 'poppies', 'porcupine', 'possum', 'rabbit', 
            'raccoon', 'ray', 'road', 'rocket', 'roses', 'sea', 
            'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 
            'snail', 'snake', 'spider', 'squirrel', 'streetcar', 
            'sunflowers', 'sweet peppers', 'table', 'tank', 'telephone', 
            'television', 'tiger', 'tractor', 'train', 'trout', 
            'tulips', 'turtle', 'wardrobe', 'whale', 'willow', 
            'wolf', 'woman', 'worm']
        
    def load_model(self):
        # load json and create model
        json_file = open('resources/cifar-100-python/100model_70.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("resources/cifar-100-python/100model_70.h5")
        print("CNN model loaded from disk")
    
    def model_compile(self):
        epochs = 50
        lrate = 0.01
        adam = Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        print(self.model.summary())
 
    def normalize(self, X_test):
        mean = 121.936
        std = 68.389
        X_test = (X_test-mean)/(std+1e-7)
        return X_test

    def model_predict(self, figure):
        test = misc.imresize(figure, (32,32,3)).transpose(2,0,1)
        test = test.astype('float32')
        test = self.normalize(test)
        test = np.expand_dims(test, axis=0)
        y_proba = self.model.predict(test)
        classed = self.model.predict_classes(test, verbose=0)
        #print(classed[0])
        if max(y_proba[0]) > 0.5:
            return(self.categories[classed[0]])
        else:
            return ''
    
    def test(self):
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
        X_test = X_test.astype('float32')
        X_test = X_test / 255.0
        # one hot encode outputs
        y_test = np_utils.to_categorical(y_test)
        scores = self.model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))