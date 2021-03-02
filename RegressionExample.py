# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:43:40 2021

@author: Bruin
"""

import pandas as pd
import numpy as np
from BasicML import Regress

df = pd.read_excel (r'C:\Users\Bruin\Desktop\Data Analysis Toolbox\Manley Data for practice\sparrows.xlsx') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
#print (df)

expl_var_names=["tl","ae","hl","ks","sv"]

X=np.array(df[expl_var_names])
Y=np.array(df["bh"])

model=Regress(X,Y,[4])
print(model.design)
