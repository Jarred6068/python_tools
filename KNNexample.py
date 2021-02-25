# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 00:03:28 2021

@author: Bruin
"""


import random as rnd
import stattools as st
import numpy as np

#make a pretend dataset
L=['a']*100+['b']*100+['c']*100
L
labels=rnd.sample(L, 100)

G=st.simnormal(100)
H=st.simnormal(100, 10, 3)
J=G+H
J=np.array(J)
J.shape
J=J.transpose()

#import KNN
from BasicML import KNN

knn1=KNN()

knn1.fit(labels, J, split=0.3)
knn1.predict()

knn1.predictedLabel
knn1.predictedValue
knn1.probabilities


#Example using Iris from skLearn
import sklearn.datasets as ds

iris = ds.load_iris()
dir(iris)

data = iris['data']
print('data', data.shape)

targets = iris['target']
print('targets', targets.shape)
#convert targets from array to character list
targets=targets.tolist()
ind=st.easySeq(len(targets))
targets=[str(targets[i]) for i in ind]

knn2=KNN(K_neighbors=10)
knn2.fit(targets, data, split=0.5)
knn2.predict()

knn2.predictedLabel
knn2.predictedValue
knn2.probabilities










