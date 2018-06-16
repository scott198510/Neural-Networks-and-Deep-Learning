# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 21:33:51 2018

@author: Scott Lee
"""

import random
import numpy as np
import mnist_loader

class Network(object):
    
    def __init__(self,sizes):
        
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        self.weights=[np.random.randn(y,x) 
             for x, y in zip(sizes[:-1],sizes[1:])]
        
#Console:import Network
        # Net=Network.Network([2,3,1])
    
    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a=sigmoid(np.dot(w,a)+b)
        return a # a is a vector with 10-dimentions

    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        if test_data:n_test=len(test_data)
        n=len(training_data)
        for j in range(epochs):
            random.shuffle(training_data) #shuffle the data randomly
            mini_batches=[training_data[k:k+mini_batch_size]
                for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data: # test_data is existed (not default none)
                print ("Epoch {0}:{1}/{2}".format(
                 j,self.evaluate(test_data),n_test))
            else: # test_data is none
                print ("Epoch{0} complete".format(j))
    
    def update_mini_batch(self,mini_batch,eta):
        Xar=[]
        Yar=[]
        for x,y in mini_batch:#maybe there are 10 (x,y) in mini_batch
            Xar.append(x)
            Yar.append(y)
        X=sum(np.array(Xar).transpose()[:]) # change 3D array to 2D array
        Y=sum(np.array(Yar).transpose()[:]) # just as 1x10x100 to 10x100
        nb,nw=self.backprop(X,Y)
        Nb=[nbx.sum(axis=1) for nbx in nb]# get the sum of mini-batch gradient
        Nb2=[nx.reshape(nx.shape[0],1) for nx in Nb]
        Nw=nw  #nw has been the sum of single nabla_w
        
        self.weights=[w-eta/len(mini_batch)*nw2 
                        for w,nw2 in zip(self.weights,Nw)]
        
        self.biases=[b-eta/len(mini_batch)*nb2 
                        for b,nb2 in zip(self.biases,Nb2)]
                
    def backprop(self,X,Y):
        nabla_b=[np.zeros((b.shape[0],X.shape[1])) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        activation=X    #initialized input x is also an initialized activated ouput
        activations=[X] # save activated ouput value 
        zs=[]
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation) 

        delta=self.cost_derivative(activations[-1],Y)*\
                sigmoid_prime(zs[-1]) # the output layer error
                
        nabla_b[-1]=delta #delta is an array with 10x15
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())
        # the result of the np.dot,means sum of single nabla_w
        for ly in range(2,self.num_layers):
            z=zs[-ly]
            sp=sigmoid_prime(z)
            delta=np.dot(self.weights[-ly+1].transpose(),delta)*sp
            nabla_b[-ly]=delta
            nabla_w[-ly]=np.dot(delta,activations[-ly-1].transpose())
        return (nabla_b,nabla_w)
        
    def evaluate(self,test_data):
        test_results=[(np.argmax(self.feedforward(x)),y)
                    for (x,y) in test_data] 
        #feedforward(x) is a 10D vector,the argmax of it,we can get the computed value
        return sum(int(x==y) for (x,y) in test_results) #get the correct forecasted number
    
    def cost_derivative(self,out_activations,y):
        return (out_activations-y)
        
def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))
        
def sigmoid_prime(z):
        return sigmoid(z)*(1-sigmoid(z))

if __name__=='__main__':
    training_data,validation_data,test_data=mnist_loader.load_data_wrapper()
    net=Network([784,100,30,10])
    net.SGD(training_data,30,10,3.0,test_data=test_data)
        
        