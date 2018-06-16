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
        self.biases =[np.random.randn(y,1) for y in sizes[1:]]
        self.weights=[np.random.randn(y,x) 
             for x, y in zip(sizes[:-1],sizes[1:])]
#Console:import Network
        # Net=Network.Network([2,3,1])
        # biases=[randn(3,1),randn(1,1)]
        # weights=[randn(3,2),randn(1,3)]
    
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
                for k in range(0,n,mini_batch_size)] #分成多个批次，每批最多mini_batch_size数据
            for mini_batch in mini_batches: #遍历每个批次数据中的每个数据
                self.update_mini_batch(mini_batch,eta)#对每个数据应用梯度下降算法
            if test_data: # test_data is existed (not default none)
                print ("Epoch {0}:{1}/{2}".format(
                 j,self.evaluate(test_data),n_test))
            else: # test_data is none
                print ("Epoch{0} complete".format(j))
    
    def update_mini_batch(self,mini_batch,eta):
        nabla_b=[np.zeros(b.shape) for b in self.biases]  # the initialized b
        nabla_w=[np.zeros(w.shape) for w in self.weights] # the initialized w
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w=self.backprop(x,y)# get the gradient for each (x,y)
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
            #对于nabla_b,nabla_w,每个（x,y)数据对计算的结果进行累计，后一个（x,y)计算结果基于前一个结果累加
        self.weights=[w-eta/len(mini_batch)*nw 
                        for w,nw in zip(self.weights,nabla_w)]
        self.biases=[b-eta/len(mini_batch)*nb 
                        for b,nb in zip(self.biases,nabla_b)]
                          
    def backprop(self,x,y):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        activation=x    # initialized input x is also an initialized activated ouput
        activations=[x] # save activated ouput value 
        zs=[]
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)
        
        delta=self.cost_derivative(activations[-1],y)*\
                sigmoid_prime(zs[-1]) # the output layer error
                
        nabla_b[-1]=delta  #nabla_b即C对b求偏导（总损失函数对b求偏导）
        nabla_w[-1]=np.dot(delta,activations[-2].transpose()) # NN&DP Page-37
        
        for l in range(2,self.num_layers):#名义上是l，实际是-l反向遍历
            z=zs[-l]  
            sp=sigmoid_prime(z)
            delta=np.dot(self.weights[-l+1].transpose(),delta)*sp
            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta,activations[-l-1].transpose()) # NN&DP Page-40
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
    