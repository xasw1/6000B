from __future__ import division, print_function, absolute_import
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize
from matplotlib import pyplot as plt
import os

# read data
train = open("/Users/yn/Downloads/data/train.txt", "r+")
val = open("/Users/yn/Downloads/data/val.txt","r+")
def process(file_):
    output = file_.split("/")
    fname = os.path.join("/Users/yn/Downloads/data/flower_photos/", output[2], output[3])
    im = imread(fname)
    image = resize(im, (32, 32 ,3))
    #imsave(fname,image)
    return image

def load_data(t):
    data = {}
    for line in t:
        values = line.split()[0]
        v = values.split('/')
        data[v[3]] = {}
        data[v[3]]['vector'] = process(values)
        data[v[3]]['class'] = line.split()[1]
    return data

testset = load_data(val)
trainset = load_data(train)

import pandas as pd
import numpy as np
test = pd.DataFrame(testset)
trainset = pd.DataFrame(trainset)

from tflearn.data_utils import shuffle, to_categorical
trainY = to_categorical(np.array(trainset.loc['class']),nb_classes=5)
testY = to_categorical(np.array(test.loc['class']),nb_classes=5)

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


network = input_data(shape=[None, 32, 32, 3], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 3)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 3)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)


network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,loss='categorical_crossentropy', name='target')


testX = []
trainX = []
for i in range(550):
    testX.append(test.loc['vector'].iloc[i])
for i in range(len(trainset.T)):
    trainX.append(trainset.loc['vector'].iloc[i])
# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': np.array(trainX)}, {'target': trainY}, n_epoch=30,
           validation_set=({'input': np.array(testX)}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='convnet_proj2')
model.save('model5_test')