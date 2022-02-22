#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
sys.path.append('../fashnMnist/')
from fashnMnist.FashnMnist import FashnMnist
from fashnMnist.Preprocessor import Preprocessor
import wandb


# In[32]:


np.random.seed(20)


# In[33]

from sklearn.model_selection import train_test_split    
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
preprocess=Preprocessor(normalization=True)
x_trainNorm, y_trainNorm, x_testNorm, y_testNorm=preprocess.Process_Fashon_mnistDataSet(x_train, y_train, x_test, y_test)
X_training, x_crossVal, y_training, y_crossval = train_test_split(x_trainNorm, y_trainNorm, test_size=0.1, random_state=20)
print(y_training.shape)
print(y_crossval.shape)
# #Load Mnist fafashion dataset using keras

# In[34]:
sweep_config = {"name": "sweep", "method": "grid"}
sweep_config["metric"] = {"name": "accuracy", "goal": "maximize"}
"""
parameters_dict = {
                "epochs": {"values": [10]},     
                "optimizer": {"values": ["rms","adam","nadam"]}, \
                "batch": {"values": [32,100]}, \
                "lr":{"values":[0.01,0.001]}, \
                "layer1_size":{"values": [32,64,128]}, \
                "layer2_size":{"values": [32,64]}, \ 
                "activation" :{"values": ["tanh","relu"]} \ 
                }
                """
parameters_dict = {
                "lossfunction": {"values": ["mse","cross"]},\
                "activation" :{"values": ["tanh"]} \
                
            
                }                
sweep_config["parameters"] = parameters_dict


# In[35]:

def train_model(config = sweep_config):
    with wandb.init(config = config):
        config = wandb.init().config
        
        #wandb.run.name = "op_{}_act_{}_lr_{}_layer1_{}_layer2_{}".format(config.optimizer,config.activation ,config.lr,config.layer1_size,config.layer2_size)
        
        wandb.run.name = "{}__{}_{}".format(config.lossfunction ,"adam",config.activation)
        model=FashnMnist(x=x_trainNorm,y=y_trainNorm,runlog=True ,lossfunction=config.lossfunction,lr=.1,epochs =3,batch=32\
                   ,weight_decay=0.001,layer1_size=20,optimizer='adam',activation=config.activation)
        """
        model=FashnMnist(x=X_training,y=y_training,  lr=config.lr,  epochs =config.epochs, batch=config.batch,                 
           layer1_size=config.layer1_size,\
           layer2_size=config.layer2_size,\
           optimizer=config.optimizer,\
           initializer="he",\
           activation=config.activation,\
           weight_decay=.5
           )
        """
        model.train()
        _,accTrain,lossTrain = model.GetRunResult(X_training,y_training)
        _,accTest,lossTest =model.GetRunResult(x_crossVal,y_crossval)
        
        wandb.log({"val_accuracy": accTest,"val_loss": lossTest, 
               "loss": lossTrain,"epochs":3, "accuracy": accTrain})
            
        
        
      
 

      


# In[36]:

import os
os.environ['WANDB_NOTEBOOK_NAME'] = 'sweepTest'
sweep_id = wandb.sweep(sweep_config, project = "cs6910_assignment1")
wandb.agent(sweep_id, function = train_model)


#




