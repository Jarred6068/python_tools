# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:31:23 2022

@author: Bruin
"""

import sys
sys.path.append('C:/Users/Bruin/Desktop/GS Academia/PhD/SEM 3 FALL 2022/Deep Learning/')
from my_example_GD_LinReg import linreg as lg
import pandas as pd
df=pd.read_csv('C:/Users/Bruin/Desktop/GS Academia/PhD/SEM 3 FALL 2022/Deep Learning/winequality-white.csv', sep = ';')

X = df.values[:, :11]
Y = df.values[:, 11]

reg = lg(X, Y, 
         optimizer = 'LSE', 
         max_iter=5000, 
         lr = 1e-5, 
         conv_crit = 0.01,
         decay=0.9,
         epochs=50,
         batchsize=64,
         method = 'SGD',
         verbose = True)
reg.split_data()
reg.fit()

# reg = lg(X, Y, optimizer = 'LSE', max_iter=5000, lr = 1e-5, conv_crit = 0.01)
# reg.split_data()
# reg.fit()

# print(reg.final_weights)
# print(reg.preds)
# print(reg.final_loss)

# reg.plot_loss()

# reg2 = lg(X, Y, optimizer = 'MSE', max_iter=5000, lr = 1e-5, conv_crit = 0.1)
# reg2.split_data()
# reg2.fit()

# print(reg2.final_weights)
# print(reg2.preds)
# print(reg2.final_loss)

# reg2.plot_loss()