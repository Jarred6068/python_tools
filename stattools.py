# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import random as rnd
import math as math
import scipy
import matplotlib.pyplot as plot


#=============================================================================
# function for the sample mean of a list
def mean(x):
    return sum(x)/len(x)

#=============================================================================
#Simple function to quickly generate sequences ranging from 0:n
def easySeq(n):
    seq=[1]*n
    for i in range(n):
        seq[i]=i
        i+=1
    return(seq)
#=============================================================================
#calculates the distance between rows (d=1) or columns(d!=1) of the numeric array 'x' 
#using one of euclidean, manhattan, or minkowski distance. For the minkowski 
#there is the option to specify q (the power parameter).
def dist(x, d=1, method="euclidean", q=3):
    nrows=x.shape[0]
    ncols=x.shape[1]
    
    if d==1:
        d=[float(1)]*(nrows**2)
        d=np.array(d)
        d=d.reshape((nrows,nrows))
        i_its=easySeq(nrows)
        j_its=easySeq(nrows)
        
        if method=="euclidean":
            p=2
            
            for i in i_its:
        
                for j in j_its:
            
                    d[i][j] = sum(abs(x[i][:]-x[j][:])**p)**(1/p)
                
        if method=="manhattan":
            p=1
            
            for i in i_its:
        
                for j in j_its:
            
                    d[i][j] = sum(abs(x[i][:]-x[j][:])**p)**(1/p)
                    
        if method=="minkowski":
            
            for i in i_its:
        
                for j in j_its:
            
                    d[i][j] = sum(abs(x[i][:]-x[j][:])**q)**(1/q)
            
                
    else:
        d=[float(1)]*(ncols**2)
        d=np.array(d)
        d=d.reshape((ncols,ncols))
        i_its=easySeq(ncols)
        j_its=easySeq(ncols)
        
        if method=="euclidean":
            p=2
            
            for i in i_its:
        
                for j in j_its:
            
                    d[i][j] = sum(abs(x[i][:]-x[j][:])**p)**(1/p)
                
        if method=="manhattan":
            p=1
            
            for i in i_its:
        
                for j in j_its:
            
                    d[i][j] = sum(abs(x[i][:]-x[j][:])**p)**(1/p)
                    
        if method=="minkowski":
            
            for i in i_its:
        
                for j in j_its:
            
                    d[i][j] = sum(abs(x[i][:]-x[j][:])**q)**(1/q)
                
    return d
#=============================================================================

# function for the sample variance of a list
def var(x):
    n=len(x)
    div=float(1/(n-1))
    errors=[float((i - mean(x))**2) for i in x]
    return div*sum(errors)

#=============================================================================
#function for scaling and centering data
def scale(X, center=True):
    
    if center == True:
        Z=[i-mean(X) for i in X]
        Z=[i/math.sqrt(var(X)) for i in Z]
        
    else:
        Z=[i/math.sqrt(var(X)) for i in Z]    
    return Z

#=============================================================================
# a function for cross tabulating two lists with a and b levels respectively:

def crosstab(X, Y):
    
    if not len(X)==len(Y):
        print("ERROR: all arguments must be same length")
        
    if not type(X) == list: 
        print("Warning: X is not a list:","type(X)=", type(X))
    
    if not type(Y) == list:
        print("Warning: Y is not a list:","type(Y)=", type(Y))
    
    nrows=len(X)
    a=list(set(X))
    b=list(set(Y))
    
    a.sort()
    b.sort()
    
    numfactors=[len(a), len(b)]
    
    
    if np.argsort(numfactors)[0]==0:
        table=[1]*(len(b))
        store2=[1]*len(a)
        
        for i in easySeq(len(b)):
            
            for j in easySeq(len(a)):
                store1=[]
                for k in easySeq(nrows):
                    
                    if X[k]==a[j] and Y[k]==b[i]:
                        store1.append(1)
                        
                store2[j]=len(store1)
            table[i]=list(store2)
                
                
    else:
        table=[1]*(len(a))
        store2=[1]*len(b)
        
        for i in easySeq(len(a)):
            
            for j in easySeq(len(b)):
                store1=[]
                
                for k in easySeq(nrows):
                    
                    if X[k]==a[i] and Y[k]==b[j]:
                        store1.append(1)
                      
                store2[j] = len(store1)
            table[i]=list(store2)
            
    return np.array(table)

#=============================================================================
#function to create a dummy/sparse-matrix representing the unique levels in 
#the factor F. F must be a list
def dummy(F):
    if(type(F)!=list):
        print("ERROR: input must be a list!")
        
    levels=list(set(F))
    L=[1]*(len(F)*len(levels))
    L=np.array(L).reshape((len(F),len(levels)))
    
    for j in easySeq(len(levels)):
        
        for i in easySeq(len(F)):
            
            if L[i,j]==levels[j]:
                L[i,j]=1
            else:
                L[i,j]=0
                
    return L


#=============================================================================
#functio to simulate normal RV's. Simulates two IID normal RV's using the 
#Box-Muller transformation
def simnormal(n, mu=0, sigma=1):
    import random as rand
    import math 
    y=[1]*n
    x=[1]*n
    yy=[i*rand.uniform(0,1) for i in y]
    xx=[i*rand.uniform(0,1) for i in x]
    a=[math.sqrt(-2*math.log(i))*math.cos(2*math.pi*j) for i,j in zip(yy,xx) ]
    b=[math.sqrt(-2*math.log(i))*math.sin(2*math.pi*j) for i,j in zip(yy,xx) ]
    c=[sigma*i+mu for i in a]
    d=[sigma*j+mu for j in b]
    return [c,d]

#=============================================================================
#function to simulate exponential RV's. Simulates exponential Random Variables 
#using the inverse CDF method
def simexp(n, lamb=1):
    import random as rand
    import math 
    x=[1]*n
    U=[i*rand.uniform(0,1) for i in x]
    E=[(-1*math.log(1-i))/lamb for i in U]
    return E

#=============================================================================
#a function to simulate Chi-Squared Random Variables (RV's). It will simulate 
#independent RV's from a normal RV such that if X_i~N(0,1)
#then Y_i=sum[x_i^2] from i =1 to n, then Y_i ~ Chi(n). 
def simChisq(n, df=1):
    sample=[1]*n
    for k in range(n):
        x=simnormal(df)
        z=[i**2 for i in x[0]]
        chi=sum(z)
        sample[k]=chi
    return sample

#=============================================================================
#a function to simulate F random variables from the ratio of two independent
#Chi-squared RV's with v1 and v2 degrees of freedom
def simF(n, V1=1, V2=1):
    for k in range(n):
        chi1=simChisq(n, df=V1)
        chi2=simChisq(n, df=V2)
        F=[(i/V1)/(j/V2) for i,j in zip(chi1,chi2)]
    return F

#=============================================================================
#a function to simulate RV's from students t-distribution with desired degrees 
#of freedom
def simt(n, df=10):
    import math as m
    nn=df+1
    sample=[1]*n
    for k in range(n):
        X=simnormal(nn)
        muhat=mean(X[0])
        sighat=var(X[0])
        sample[k]=(muhat)/(m.sqrt(sighat)/m.sqrt(nn))
    return sample

#=============================================================================
#a function to calculate the CDF probability of a given value from students t
#distribution using Monte Carlo Integration
def probt(t, df):
    n=100
    b=100
    p_list=[1]*b
    for k in range(b):
        T=simt(n, df=df)
        p_list[k]=sum([i<=t for i in T])/n
    prob=mean(p_list)
    return round(prob, ndigits=5)

#=============================================================================
#a function to calculate the CDF probability of a given value from the F 
#distribution using Monte Carlo Integration 
def probf(f, df1, df2):
    n=100
    b=100
    p_list=[1]*b
    for k in range(b):
        F=simF(n, V1=df1, V2=df2)
        p_list[k]=sum([i<=f for i in F])/n
    prob=mean(p_list)
    return round(prob, ndigits=5)

#=============================================================================
#a function to calculate the CDF probability of a given value from the normal 
#distribution using Monte Carlo Integration 
def probnorm(x, mu=0, sigma=1):
    n=1000
    b=100
    p_list=[1]*b
    for k in range(b):
        Z=simnormal(n, mu=mu, sigma=sigma)
        p_list[k]=sum([i<=x for i in Z[0]])/n
    prob=mean(p_list)
    return round(prob, ndigits=5)

#=============================================================================
#a function to calculate the CDF probability of a given value from the 
#Exponential distribution using Monte Carlo Integration 
def probexp(x, lam=1):
    n=1000
    b=100
    p_list=[1]*b
    for k in range(b):
        E=simexp(n, lamb=lam)
        p_list[k]=sum([i<=x for i in E])/n
    prob=mean(p_list)
    return round(prob, ndigits=5)

#=============================================================================
#a function to calculate the CDF probability of a given value from the 
#Chi-Squared distribution using Monte Carlo Integration
def probchi(x, df=1):
    n=1000
    b=100
    p_list=[1]*b
    for k in range(b):
        E=simChisq(n, df=df)
        p_list[k]=sum([i<=x for i in E])/n
    prob=mean(p_list)
    return round(prob, ndigits=5)

#=============================================================================





















