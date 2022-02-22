import numpy as np
from fashnMnist.NeuralNetwork import NeuralNetwork
from fashnMnist.optimizer.MomentumGradiantDecent import MomentumGradiantDecent
from fashnMnist.optimizer.RmsProp import RmsProp
from fashnMnist.optimizer.NAG import NAG
from fashnMnist.optimizer.Adam import Adam
from fashnMnist.optimizer.NAdam import NAdam


class FashnMnist:
    def __init__(self,x
                 , y
                 ,lr = .1
                 ,epochs =100
                 ,batch=32
                 ,HiddenLayerNuron=[32,64,10]
                 ,decay_rate=0
                 ,activation='tanh'
                 ,optimizer='mgd'
                 ,beta1=0.9
                 ,beta2=0.99
                 ,gamma=0.9
                 ,beta=.9
                 ,initializer='he'
                 ,dropout_rate=0
                 ,weight_decay=0
                 ,layer1_size=0
                 ,layer2_size=0
                 ,layer3_size=0
                 ,layer4_size=0
                 ,layer5_size=0
                 ,runlog=True
                 ,lossfunction='cross'
                 ,wandb=None
                 ,wandbLog=False,
                 x_val=None
                 ,y_val=None):
                
                self.network=None
 
                self.y=y
                self.optimizer=optimizer
                
                
                hiddenLayers=[]
                if(layer1_size>0):
                    hiddenLayers.append(layer1_size)
                    if(layer2_size>0):
                        hiddenLayers.append(layer2_size)
                        if(layer3_size>0):
                            hiddenLayers.append(layer3_size)
                            if(layer4_size>0):
                                hiddenLayers.append(layer4_size)
                                if(layer5_size>0):
                                    hiddenLayers.append(layer5_size)
                    
                    
                    hiddenLayers.append(y.shape[1])
                else:
                    hiddenLayers=   HiddenLayerNuron 
                if (self.optimizer=='gd'):
                    self.network=NeuralNetwork( x=x, y=y, lr = lr, wandb=wandb,wandbLog=wandbLog,x_val=x_val,y_val=y_val, epochs =epochs,batch=batch,HiddenLayerNuron=hiddenLayers,activation=activation,decay_rate=decay_rate,initializer=initializer,weight_decay=weight_decay,runlog=runlog,lossfunction=lossfunction)
                    
                if (self.optimizer=='sgd'):
                    self.network=NeuralNetwork( x=x, y=y, lr = lr, wandb=wandb,wandbLog=wandbLog, x_val=x_val,y_val=y_val,  epochs =epochs,batch=1,HiddenLayerNuron=hiddenLayers,activation=activation,initializer=initializer,weight_decay=weight_decay,runlog=runlog,lossfunction=lossfunction)
                    
                if (self.optimizer=='mgd'):
                    self.network=MomentumGradiantDecent( x=x, y=y, lr = lr, wandb=wandb,wandbLog=wandbLog,x_val=x_val,y_val=y_val,   epochs =epochs,batch=batch,HiddenLayerNuron=hiddenLayers,activation=activation,dropout_rate=dropout_rate,decay_rate=decay_rate,initializer=initializer,weight_decay=weight_decay,runlog=runlog,lossfunction=lossfunction)
                    
         
                if(self.optimizer=='nag'):
                     self.network=NAG( x=x, y=y, lr = lr, wandb=wandb,wandbLog=wandbLog, x_val=x_val,y_val=y_val,  epochs =epochs,batch=batch,HiddenLayerNuron=hiddenLayers,activation=activation,gamma=gamma,initializer=initializer,weight_decay=weight_decay,dropout_rate=dropout_rate,runlog=runlog,lossfunction=lossfunction)
                
                if(self.optimizer=='adam'):
                    self.network=Adam( x=x, y=y, lr = lr,  wandb=wandb,wandbLog=wandbLog, x_val=x_val,y_val=y_val, epochs =epochs,batch=batch,HiddenLayerNuron=hiddenLayers,activation=activation,decay_rate=decay_rate,beta1=beta1,beta2=beta2,initializer=initializer,dropout_rate=dropout_rate,weight_decay=weight_decay,runlog=runlog,lossfunction=lossfunction)
                    
                if(self.optimizer=='nadam'):
                    self.network=NAdam( x=x, y=y, lr = lr,  wandb=wandb,wandbLog=wandbLog, x_val=x_val,y_val=y_val, epochs =epochs,batch=batch,HiddenLayerNuron=hiddenLayers,activation=activation,decay_rate=decay_rate,beta1=beta1,beta2=beta2,initializer=initializer,dropout_rate=dropout_rate,weight_decay=weight_decay,runlog=runlog,lossfunction=lossfunction)
                
                if(self.optimizer=='rms'):
                     self.network=RmsProp( x=x, y=y, lr = lr,  wandb=wandb,wandbLog=wandbLog, x_val=x_val,y_val=y_val, epochs =epochs,batch=batch,HiddenLayerNuron=hiddenLayers,activation=activation,decay_rate=decay_rate,initializer=initializer,dropout_rate=dropout_rate,weight_decay=weight_decay,runlog=runlog,lossfunction=lossfunction)
                    
                
    def train(self): 
        self.network.train()
       
    def predict(self,x,y):
        prediction=self.network.getResults(x)
        pred=y-prediction
        accurecy=sum(x==0 for x in pred)
        accurecy=accurecy/len(pred)
        print('Test accuracy='+str(accurecy*100))
        return prediction
    
    def TrainingStatistics(self):
        return self.network.runAccurecy,self.network.runLoss
        
        
    def GetRunResult(self,x,y):
        prediction,acc,loss=self.network.getResults(x,y)
        return  prediction,acc,loss
    
    