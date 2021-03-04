# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:43:40 2021

@author: Bruin
"""

import pandas as pd
import numpy as np
from BasicML import Regress

#Manley data "sparrow survival data"
df = pd.read_excel (r'C:\Users\Bruin\Desktop\Data Analysis Toolbox\Manley Data for practice\sparrows.xlsx') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
#print (df)

expl_var_names=["tl","ae","hl","ks","sv"]

X=np.array(df[expl_var_names])
Y=np.array(df["bh"])

model=Regress(X,Y,[4], interactions=False)
print(model.design_matrix)

model.fit()
import matplotlib.pyplot as p
p.plot(model.predicted, model.residuals, 'bo')+p.hlines(0, 30, 33)

#iris dataset

df2 = pd.read_excel (r'C:\Users\Bruin\Desktop\Data Analysis Toolbox\Manley Data for practice\iris.xlsx')

expl_var_names2=['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width',
       'Species']

X2=np.array(df2[expl_var_names2[1:len(expl_var_names2)]])
Y2=np.array(df2[expl_var_names2[0]])