# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:49:40 2021

@author: Jarred M. Kvamme
"""
#import dependencies
import numpy as np
import stattools as st
import random as rnd
import math as m

#KNN algorithm:
    
class KNN:
    """implementation of the K-nearest Neighbor Algorithm"""
    """with choice of distance metric"""
    
    def __init__(self, K_neighbors=5):
        """Two input variables are: the number of neighbors considered"""
        """ the type of distance metric used"""
        
        #save property
        self.K=K_neighbors
        
    def fit(self, labels, Data, split=0.5):
        """input argument Data should be of type numpy.array()"""
        """input arg lables should be character list"""
        """input arg split = % training/validation partition"""
        
        #save properties
        self.split=split
        self.Nrows=Data.shape[0]
        self.labels=labels
        #create indexing var
        rowIND=st.easySeq(self.Nrows)
        
        self.trainR=rnd.sample(range(self.Nrows),int(round(self.split*self.Nrows)))
        for i in self.trainR:
            rowIND.remove(i)
        self.testR=rowIND
        
        #pre-allocate
        L1=list(labels)
        testL=[]
        trainL=[]
        for i in self.testR:
            testL=testL+list(L1[i])
        for i in self.trainR:
            trainL=trainL+list(L1[i])
            
        #save properties
        self.trainLabels=trainL
        self.testLabels=testL
        self.training=Data[self.trainR][:]
        self.testing=Data[self.testR][:]
        
    def predict(self, dist_type="euclidean"):
        
        """(1) Calculate the testing set predictions based on the"""
        """training set distances to i'th Observation and (2) store"""
        """neighbors and (3) designate label by popular vote"""
        """input arg dist_type is metric to be used"""
        
        #save property
        self.metric=dist_type
        #set iterative index
        ind=st.easySeq(len(self.testR))
        #allocation
        training=self.training.tolist()
        testing=self.testing.tolist()
        
        #pre-allocate
        save_list=[1]*len(ind)
        neighborsIDX=[1]*len(ind)
        neighbors=[1]*len(ind)
        neighborLabels=[1]*len(ind)
        
        #find neighbors and their values
        for i in ind:
            mat=np.array(list(training+[testing[i]]))
            disti=st.dist(mat, method=dist_type)
            save_list[i]=disti[disti.shape[0]-1][:].tolist()
            neighborsIDX[i]=np.argsort(save_list[i]).tolist()[1:(self.K+1)]
            neighbors[i]=self.training[neighborsIDX[i]][:].tolist()
        
        #obtain class labels of neighbors
        for j in ind:
            neighborLabels[j]=[self.trainLabels[k] for k in neighborsIDX[j]]
        
        #save property
        self.neighbors=np.array(neighbors)
        self.neighborLabels=neighborLabels
        #set iterative index
        ind2=st.easySeq(self.neighbors.shape[0])
        #pre-allocate
        predicted=[1]*self.neighbors.shape[0]
        
        #calculate predicted values as (1 X p) mean vec (column means)
        for i in ind2:
            predicted[i] = np.mean(self.neighbors[i], axis=0).tolist()
            
        #save property
        self.predictedValue=np.array(predicted)
        
        #pre-allocate
        probabilities=[1]*len(ind2)
        predictedLabel=[1]*len(ind2)
        #get unique classes
        Classes=list(set(self.labels))
        #get prediction probabilities and predicted class labels
        for i in ind2:
            probabilities[i]=[self.neighborLabels[i].count(j)/self.K for j in Classes]
            predictedLabel[i]=Classes[np.argmax(probabilities[i])]

        #save properties to object
        self.probabilities=probabilities
        self.predictedLabel=predictedLabel
            
            
            
            
            
            
            
            
            
        