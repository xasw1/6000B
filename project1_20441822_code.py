import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import csv

# read data
f = open('/Users/yn/Downloads/traindata.csv','rb')
trainingdata = pd.read_csv(f,header=None)
f = open('/Users/yn/Downloads/trainlabel.csv','rb')
label_set = pd.read_csv(f,header=None)
f = open('/Users/yn/Downloads/testdata.csv','rb')
test = pd.read_csv(f,header = None)

# data description
print(set(label_set[0]))
print(label_set[label_set == 1].sum()/len(label_set))

# change type
dataset = []
for i in range(len(trainingdata)):
    dataset.append(np.array(trainingdata.iloc[i]))
testset = []
for i in range(len(test)):
    testset.append(np.array(test.iloc[i]))
c,r = label_set.shape
label_set = np.array(label_set).reshape(c,)

# compare performance
## CV-10
# decision tree
clf = DecisionTreeClassifier()
scores = cross_validation.cross_val_score(clf, np.array(dataset), np.array(label_set), cv=10)
print('Tree:\n10-cv:',scores,'\nmean:',scores.mean(),'\nvar:',scores.var())
# randomforest
clf = RandomForestClassifier()
scores = cross_validation.cross_val_score(clf, dataset, label_set, cv=10)
print('RandomForest:\n10-cv:',scores,'\nmean:',scores.mean(),'\nvar:',scores.var())
# neural_network
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier()
scores = cross_validation.cross_val_score(clf, dataset, label_set, cv=10)
print('neural network:\n10-cv:',scores,'\nmean:',scores.mean(),'\nvar:',scores.var())
# Adaboost
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_validation.cross_val_score(clf, dataset, label_set, cv=10)
print('Adaboost:\n10-cv:',scores,'\nmean:',scores.mean(),'\nvar:',scores.var())
# LogisticRegression
clf = LogisticRegression()
scores = cross_validation.cross_val_score(clf, dataset, label_set, cv=10)
print('LogisticRegression:\n10-cv:',scores,'\nmean:',scores.mean(),'\nvar:',scores.var())
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=100, max_depth=52, random_state=0)
scores = cross_validation.cross_val_score(clf, dataset, label_set, cv=10)
print('ExtraTreesClassifier:\n10-cv:',scores,'\nmean:',scores.mean(),'\nvar:',scores.var())

# predict
clf = ExtraTreesClassifier(n_estimators=100, max_depth=52, random_state=0)
clf = clf.fit(dataset,label_set)
result = clf.predict(test)
