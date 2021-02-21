# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 00:03:28 2021

@author: Jarred M. Kvamme
"""

#make a pretend dataset
import random as rnd
import stattools as st
import numpy as np

L=['a']*100+['b']*100+['c']*100
L
labels=rnd.sample(L, 100)

G=st.simnormal(100)
H=st.simnormal(100, 10, 3)
J=G+H
J=np.array(J)
J.shape
J=J.transpose()


from BasicML import KNN

knn1=KNN()

knn1.fit(labels, J, split=0.3)
knn1.predict()

knn1.predictedLabel
knn1.predictedValue
knn1.probabilities
