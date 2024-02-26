# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:09:49 2022

@author: Bruin
"""

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras # you may need to install tensorflow using 'pip install tensorflow'
from sklearn.model_selection import train_test_split

## load the digits dataset
def load_digits(show_sample = True):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    #show first 100 images
    if show_sample == True:
        nImg = 4
        for i in range(nImg*nImg):
            plt.subplot(nImg, nImg, i+1)
            plt.imshow(x_train[i], cmap = 'Greys_r')
        plt.show()
        
    x_train_1 = np.reshape(x_train, [x_train.shape[0], x_train.shape[1] * x_train.shape[2]])
    x_test_1 = np.reshape(x_test, [x_test.shape[0], x_test.shape[1] * x_test.shape[2]])
    x_train_2 = x_train_1/255
    x_test_2 = x_test_1/255
    
    return x_train_2, y_train, x_test_2, y_test

x_train, y_train, x_test, y_test = load_digits()
print(x_train.shape, x_test.shape)

y_train_onehot = keras.utils.to_categorical(y_train)
y_test_onehot = keras.utils.to_categorical(y_test)


## Task 1: Activation functions: implement the softmax function. 5 points
def sigm(z):
    return 1/(1 + np.exp(-z))

def dsigm(z):
    return sigm(z)*(1 - sigm(z))

def softmax(z):
    ''' softmax function for the output layer.
        The softmax function should be able to work if batch size is greater than 1.

        parameters:
            z: input numpy array. m_batch * 10
        
        return: m_batch * 10     
    '''
    ## add your code here
    e_z = np.exp(z - np.max(z, axis =  0)) 
    norms = e_z.sum(axis = 1)[:,None]
    return e_z / norms


# def dsoftmax(z):
#     z_shape = z.shape[1]
#     dz = np.zeros([z_shape, z_shape])
#     for j in range(z_shape):
#         for i in range(z_shape):
            
#             if i == j:
#                 dz[i,j] = np.dot(z[:,i],(1-z[:,i]))
#             else:
#                 dz[i,j] = np.dot(-z[:,i],z[:,j])
                
    
#     return dz
            
            

    ##
    
# test the softmax function
z = np.array([[1, 2, 3], [4, 2, 4]])
s = softmax(z)
print(s)



class Layer:
    
    def __init__(self, units, input_dim, activation = None): 
        '''creat a layer and initialize weights and bias. This is the constructor that 
          will be executed automatically when we create a layer instance. 
            
            parameters:
                units: the number of hidden nodes
                input_dim: dimensionality of the layer input
                activation: activation function
                
        '''
        
        self.units = units
        self.activation = activation
        self.input_dim = input_dim
        
        #initialize weights and bias, and their gradients. 10 points.
        #check here: https://www.deeplearning.ai/ai-notes/initialization/
        ## add your code here
        #------------------------
        np.random.seed(0)        

        self.W = np.random.normal(loc = 0, scale = 1/input_dim, size = (input_dim, units))
        self.bias = np.random.normal(loc = 0, scale = 1/input_dim, size = (units,))
        
        #initilize gradients of weights(gW) and bias(gBias)
        self.gW = np.zeros((input_dim, units))
        self.gBias = np.zeros((units,))
        
        #-----------------------
        
    def run(self, inputs):
        ''' calculate the net input and activation output of the current layer
        
            inputs: layer input. (n_sample * n_features)
          
            return:
                self.output. the activation output
        '''
        
        ## add your code here
        #-------------------------------------------
        #calculate the net input. 5 points
        self.inputs = inputs
        self.net = np.add(np.dot(inputs, self.W),self.bias)
       
        #calculate activation output. 10 points
        #should deal with both softmax and sigmoid activations
        #use self.activation to choose the activation function of the current layer
        
        if self.activation == 'sigm':
            self.output = sigm(self.net)
            
            
        if self.activation == 'softmax':
            self.output = softmax(self.net)
            
            
        #-----------------------------------------
        
        return self.output

## Test you code to create an NN with two layers 
# create a layer with 20 hidden nodes
L1 = Layer(units = 20, input_dim = 784, activation = 'sigm')
print('L1:', L1.input_dim, L1.units, L1.activation)

# print out the setup of the second layer
h1 = L1.run(x_train)
print('h1:', h1.shape)















## Task 3: complete the following NN class. 60 points
class NeuralNetwork:
    
    def __init__(self):
        self.layers=[] # list of layers
        
    # Task 3.1: implement the 'add' function. 5 points     
    def add(self, units, input_dim, activation = 'sigm'):
        '''add one layer to neural network
        
            parameters:
                units: the number of nodes of current layer
                input_dim: input dimension (the number of nodes of the previous layer)
                activation: the activation function
        '''
        
        ## add your code here
        # create and append a new layer the self.layers 
        #
        self.layers.append(Layer(units = units, 
                                  input_dim = input_dim, 
                                  activation=activation))
        
        ##
        
    # Task 3.2: implement the cross-entropy loss. 5 points   
    def loss(self, y_pred, y):
        '''loss function: 1/n_samples*sum_samples(sum_output(-y_k*log(y_pred_k)))
            
            parameters:
                y_pred: predictions(n_samples * 10)
                y: target(one-hot vectors: n_samples * 10)
            return:
                loss
        '''
        
        m = y.shape[0] # the number of samples
        
        ## add your code here  
        
        loss = (1/m)*np.sum(np.sum(-y * np.log(y_pred+1e-16), axis = 1))
        
        ##
        
        return loss
    
    # Task 3.3: implement the forward propagation process. 5 points.
    def forward_prop(self, inputs):
        '''forward propagation calculates net input and output for all layers
            
            parameters:
                inputs: input data(n_samples * n_features)
            
            return:
                out: the output of the last layer
            
            Tip: call the run function layer by layer
        '''
        
        nLayers = len(self.layers)
        
        ## add your code here
        for i in range(nLayers):
            out =  self.layers[i].run(inputs)
            inputs = out
        ##
        
        return out
    
    # Task 3.4: implement the prediction function. 5 points
    # tip: using np.argmax to convert onehot vectors to categorical values
    def predict_class(self, x):
        '''predict class lables (0, 1, 2, 3, ..., 9) for data samples
        
            parameters:
                x: input(n_samples * n_features) 
            return:
                class labels
        '''
        ##add your code here
        pred = self.forward_prop(inputs = x)
        pred_labels = np.argmax(pred, axis = 1)
        
        return pred_labels
        #  
        ##
        
    # Task 3.5: complete the following 'train' function. 10 points.
    def train(self, inputs, targets, lr = 0.001, batch_size = 32, epochs = 50):
        '''implement the SGD process and use Back-Propagation algorithm to calculate gradients 
            
            parameters:
                inputs: training samples
                targets: training targets
                lr: learning rate
                batch_size: batch size
                epochs: max number of epochs
                
            return:
                loss_hist
        '''
        
        m = len(targets)  
        y_true = targets
        x_true = inputs
        #print(m, targets.shape)
        self.loss_hist = np.zeros(epochs)
        self.epoch_losses = []
        self.batch_losses = []
        
        for i in range(epochs):
            #shuffle the data
            idx = np.arange(m)
            np.random.shuffle(idx)
            inputs = inputs[idx]
            targets = targets[idx]
            
            for b in range(int(m/batch_size)):
                b_start= b*batch_size
                b_end = min((b+1)*batch_size, m)
                
                x_batch = inputs[b_start:b_end, :]
                y_batch = targets[b_start:b_end, :]
                
                ## add your code here
                
                # 1. run forward propagation using the current batch to 
                #    calculate net input and output for all layers
                out = self.forward_prop(inputs = x_batch)
                
                # 2. call BP to calculate all gradients (gW and gBias)
                self.BP(x_batch, y_batch)
               
                # 3. update all weights and bias
                self.updateWeights(lr)
                
                self.batch_losses.append(self.loss(out, y_batch))
                
                
            self.epoch_losses.append(np.mean(self.batch_losses))
                ##
                
            lr = lr*0.95
            
            ## add your code here
            
            # 4. calculate and record the loss of current epoch
            y_pred = self.forward_prop(inputs = x_true)
            l = self.loss(y_pred, y_true)
            self.loss_hist[i] = l
            print('current loss = {}'.format(self.loss))
            
            # 5. print out the loss of current epoch
            
            ##
            
        return self.loss_hist
    
    
    def get_doj_dnetk_jacobian(self):
        out_dim = self.layers[1].units
        jacob = np.zeros((out_dim, out_dim))
        for k in range(out_dim):
            o_k = self.layers[1].output[:, k]
            for j in range(out_dim):
                o_j = self.layers[1].output[:, j]
                if j == k:
                    jacob[j, k] = np.dot(o_k, 1-o_k)
                else:
                    jacob[j, k] = np.dot(-o_j, o_k)
                    
        return jacob
                
                
   
    #Task 3.6: implement the BP algorithm. 20 points
    def BP(self, x, y):
        ''' Back-propagation algorithm. The implementation should be able to calculte
            gradients for a neural network with at least 3 layers.
            
            parameters:
            x: input samples (n_samples * n_features)
            y: onthot vectors (n_samples * 10)
            
        '''
        
        nLayers = len(self.layers)
        mbatch = x.shape[0]
        
        delta_k_store = []
        delta_j_store = []
        #for L in reversed(range(nLayers)):
        O = self.layers[1].output
        #start by iterating over output nodes
        for k in range(self.layers[1].units):
            #storage
            dl_dnetk = np.zeros(self.layers[1].units)
            #again iterate over output nodes
            dl_doj = np.multiply(-y, 1/(mbatch*O)).sum()
            #get derivative of oj/netk the jacobian
            doj_dnetk = self.get_doj_dnetk_jacobian()
            #store                     
            dl_dnetk = dl_doj*doj_dnetk[:,k]
                #sum over classes and store to list    
            delta_k_store.append(dl_dnetk)
            self.layers[1].gW[:, k] = (delta_k_store[k]*self.layers[1].inputs).sum(axis=0)
            self.layers[1].gBias[k] = delta_k_store[k]

                
        for h in range(self.layers[1].input_dim):
            mkj_delta_k = np.inner(np.array(delta_k_store),self.layers[1].W[h,:])
            #derivative of hidden layer activation
            g_h_prime = dsigm(self.layers[1].output)
            #     m_kj_delta_k = (m_j*delta_k_store).sum()
            #     #calculate delta j sum over classes
            dl_dnetj = mkj_delta_k*g_h_prime
            #     #sum over classes and store
            delta_j_store.append(dl_dnetj)
        
            #     # 1. calculate gradients for the hidden-to-output layers. 10 points 
            #     for i in range(self.layers[1].input_dim):
            #         xi = x[:, i]
            #         self.layers[0].gW[i,h] = np.inner(delta_j_store[h], xi)
            #         self.layers[0].gBias[i] = delta_j_store[h].sum()
                # 2. calculate gradients for the input-to-hidden layers. 10 points  
        
        
        
        
                

        
            
    #Task 3.7: update all weights and bias. 5 points            
    def updateWeights(self, lr):
        '''
            parameters:
                lr: learning rate
                
        '''
        nLayers = len(self.layers)
        
        ##add your code here
        for i in range(nLayers):
            self.layers[i].W  = self.layers[i].W - lr*self.layers[i].gW
            self.layers[i].bias = self.layers[i].bias - lr*self.layers[i].gBias
        ##
            
    # Task 3.8: calculate the accuracy. 5 points           
    def Acc(self, y, y_pred):
        '''calculate accuracy
        
            parameters:
                y: target: categorical values (0, 1, ...9). n_samples * 1
                y_pred: prediction: 0,1,2, ..9. n_samples *1
                
            return: acc
        '''
        
        ##add your code here
        
        ##
        
        
        
        
        
        
        
        
        
        
        
        
# Task 4: Evaluation

# create a 3-layer NN. 5 points
##add you code here
#------------------------------------
nn = NeuralNetwork()
nn.add(20, 784, 'sigm')
nn.add(10, 20, 'softmax')
nn.forward_prop(x_train)

nn.train(x_train, y_train_onehot, lr = 0.0001, epochs=5, batch_size=64)

#dsoftmax(nn.layers[1].net)
#------------------------------------


# 2. train the NN.  5 points
## add you code here
#------------------------------------
# transform y_train to onehot vectors


# train the network


#------------------------------------


# 3. calculte and print out the test and training accuracy. 5 points
##add you code here
#---------------------------------------------

#------------------------------------------------
