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
import pandas as pd

#--------------------------->>>KNN algorithm<<<-------------------------------
    
class KNN:
    """implementation of the K-nearest Neighbor Algorithm"""
    """with choice of distance metric"""
    
    def __init__(self, K_neighbors=5):
        """input variable: the number of neighbors considered"""
        
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
    
    def performance(self):
        #obtain the confusion matrix using stattools
        self.ConfusionMatrix=st.crosstab(self.predictedLabel, self.testLabels)
        
        #get total, rowsums, colsums, and correct predictions (diagonals)
        total=sum(np.sum(self.ConfusionMatrix, axis=0))
        rowsum=np.sum(self.ConfusionMatrix, axis=1)
        colsum=np.sum(self.ConfusionMatrix, axis=0)
        correct=sum(np.diag(self.ConfusionMatrix))
        
        #calculate ERROR
        self.Error=1-(correct/total)
        
        #calc observed expected and cohens kappa coefficient
        observed=1-self.Error
        expected=sum( (rowsum/total)*(colsum/total) )
        self.Kappa=(observed-expected)/(1-expected)
           
        
        
        
#-------------------------->>>Regression Algorithm<<<-------------------------
class Regress:
    """implementation of Regression"""
    """Continuous Prediction Procedure:"""
    
    def __init__(self, X, Y, idx, interactions=True):
        """ initialization of data attributes in "data" """
        
        """ X should be an np.array where each row represents the
            values of the explanatory variable(s) for the i{th} 
            observation
        """
        
        """idx should contain the column indexes of X that are multi-level
            factors
        """
        #obtain the design matrix for the model and store pieces
        self.design_matrix=st.model_matrix(X, idx, interactions)
        self.Y_obs=Y
        self.K=self.design_matrix.shape[1]-1
        self.n=X.shape[0]
        self.model_df=X.shape[0]-self.design_matrix.shape[1]
        
        #obtain the total number of interaction coefficients
        #(used for hypothesis testing and anova calculations)
        if interactions==True:
            num_fac=len(idx)
            print(num_fac)
            levels=[0]*1000
            for i in idx:
                levels[i]=len(set(X[:,i]))-1
            
            num_interact=(X.shape[1]-num_fac)*sum(levels)
            print(num_interact)
            self.num_interactions=num_interact
        
        
    def fit(self):
        """fits the linear model by least squares"""
        """Key model aspects are stored"""
        #compute the model by least-squares fit
        XX=np.dot(self.design_matrix.transpose(), self.design_matrix)
        XX_inv=np.linalg.inv(XX)
        XX_invX=np.dot(XX_inv, self.design_matrix.transpose())
        self.XX_inv=XX_inv
        
        #store model information and parameter estimates
        self.coefficients=np.dot(XX_invX, self.Y_obs)
        self.predicted=np.dot(self.design_matrix, self.coefficients)
        self.residuals=self.Y_obs-self.predicted
        self.SSE=np.dot(self.residuals.transpose(), self.residuals)
        self.SSR=sum((self.predicted-st.mean(self.Y_obs))**2)
        self.Error_variance=self.SSE/self.model_df
        self.beta_cov=self.Error_variance*self.XX_inv
        bcov_list=np.diag(self.beta_cov).tolist()
        self.beta_SE=np.array([m.sqrt(i) for i in bcov_list])
        
        
    def T_test(self):
        
        """tests the coefficients of the linear model"""
        
        #preallocate space
        T_observed=[0]*(self.K+1)
        P_value=[0]*(self.K+1)
        names=["Beta_"]*(self.K+1)
        iters=st.easySeq(self.K+1)
        iters=[str(i) for i in iters]
        row_names=["A"]*(self.K+1)
        col_names=["T_obs", "P.value","SE"]
        
        #calculate test stats and pvalues
        for i in st.easySeq(self.K+1):
            T=self.coefficients[i]/self.beta_SE[i]
            T_observed[i]=T
            P=2*(1-st.probt(abs(T), self.model_df))
            P_value[i]=P
            col=[names[i]+iters[i]]
            row_names[i]=" ".join(col)
        
        tab=[T_observed]+[P_value]+[self.beta_SE.tolist()]
        tab=np.array(tab).transpose().tolist()
        
        #output summary table
        self.summary=pd.DataFrame(tab, row_names, col_names)
        print("Model Summary and Coefficient Estimates:")
        print(self.summary)
        
        
    def anova(self, null_hypoth=[], hypoth_matrix=[], test_interact=True):
        
        """tests on the coefficients and submodels"""
        
        """if interactions_test=TRUE all interaction
           submodels ares tested
        """
        """user has the option to specify specific contrasts
           by inputing the null hypothesis vector and general 
           hypothesis matrix
        """
        # #if a specific contrast matrix or hypothesis is not specified 
        # #conduct general linear hypothesis tests
        # if not hypoth_matrix:
        #     hypoth_matrix=np.diag([1]*len(self.coefficients))
            
        
        # if not null_hypoth:
        #     print("H0: Beta = 0")
            
        #     LB=np.dot(hypoth_matrix, self.coefficients)
        #     C1=np.dot(hypoth_matrix, self.XX_inv)
        #     C2=np.dot(C1, hypoth_matrix.transpose)
            
        #     numer=np.dot(np.dot(LB,C2),LB.transpose())
        #     denom=np.sum()
            
        
        
        
        

        

        
        
        
        
            
            
            
            
            
        