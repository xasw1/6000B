from __future__ import division, print_function, absolute_import
from skimage.io import imread                                
from skimage.io import imsave
from skimage.transform import resize
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
# read data
testset = open("/Users/yn/Downloads/data/test.txt", "r+")
#al = open("/Users/yn/Downloads/data/val.txt","r+")
def process(file_):
    output = file_.split("/")
    fname = os.path.join("/Users/yn/Downloads/data/flower_photos/", output[2], output[3])
    im = imread(fname)
    image = resize(im, (32, 32, 3))
    #aimsave(fname,image)
    return image

def load_data(t):
    data = {}
    for line in t:
        values = line.split()[0]
        v = values.split('/')
        #data[v[3]] = {}
        data[v[3]] = process(values).reshape([32,32,3])
        #if line.split()[1] == 
        #data[v[3]]['class'] = int(line.split()[1])
    return data

test = load_data(testset)
test = pd.Series(test)
testX = []
for i in range(len(test)):
    testX.append(test.iloc[i])

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
model = tflearn.DNN(network, tensorboard_verbose=0)
model.load('model2_test')
#model = Evaluator(network)
result = model.predict(np.array(testX))

re = []
for i in range(len(result)):
    re.append(list(result[i]).index(max(result[i])))    

import csv
csvFile = open('/Users/yn/Desktop/project2_20441822.csv', "w")
writer = csv.writer(csvFile)
for line in re:
    writer.writerow([line])