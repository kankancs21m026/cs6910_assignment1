import numpy as np
import sys
sys.path.append('../fashnMnist/')
from fashnMnist.NeuralNetwork import NeuralNetwork

class NAG(NeuralNetwork):
    def __init__(self, x, y, lr = .5, wandb=None,wandbLog=False, x_val=None,y_val=None,epochs =100,HiddenLayerNuron=[60,10],activation='tanh',
                 beta=0.9,gamma=0.8,batch=32,initializer='he',weight_decay=0,dropout_rate=0,runlog=True,lossfunction='cross'):
                
          
                # invoking the __init__ of the parent class 
                NeuralNetwork.__init__(self, x, y, lr = lr, wandb=wandb,wandbLog=wandbLog, x_val=x_val,y_val=y_val,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,gamma=gamma,weight_decay=weight_decay,dropout_rate=dropout_rate,runlog=runlog,lossfunction=lossfunction)
                
                
          
    def train(self):
        totalLayer=len(self.HiddenLayerNuron)
        
        #initialize all parameters
        if(self.runlog):
            print('Starting NAG')
            print('.....................................')
        prev_w,prev_b  = self.DW, self.DB
        for epoch in range(self.epochs):
            gamma= self.momentumUpdate(epoch+1)
            
            
            
            self.lr=self.controlLearningRate(epoch,self.epochs)
                
            for i in range(0, self.x.shape[0], int(self.batch)):
                    #update weight and bias
                self.resetWeightDerivative()
                
                
                
                vw=self.W
                vb=self.b
                for w in range(totalLayer):
                    self.W[w]=self.W[w]-gamma*prev_w[w]
                    self.b[w]=self.b[w]-gamma*prev_b[w]
                
                self.xBatch =self.x[i:i+self.batch]
                self.yBatch  = self.y[i:i+self.batch]
                pred=self.feedforward()
                self.backprop()
                
                self.W=vw
                self.b=vb
                    
             
                prev_w,prev_b=self.updateParam( gamma,prev_w,prev_b)
               
                #verify loss after each epoch
            self.xBatch = self.x
            self.yBatch  =self.y
           
            pred=self.feedforward()
            
            acc=self.accurecy(pred,self.yBatch)
            loss=self.calculateLoss()
            self.runAccurecy.append(acc)
            self.runLoss.append(loss)
            
            #print details 
            self.printDetails(epoch,self.epochs,acc,loss)
        if(self.runlog):
            print()
            print('Completed')
            print('.....................................')
        
    def updateParam(self, gamma,prev_w,prev_b): 
        totalLayer=len(self.HiddenLayerNuron)
        
       
        for i in range(totalLayer):
           
            vw= gamma*(prev_w[i])+(self.lr)*(self.DW[i] )
            vb= gamma*(prev_b[i])+(self.lr)*(self.DB[i] )
            
          
            self.W[i] = self.W[i] - vw
            self.b[i] = self.b[i] - vb
            
            prev_w[i],prev_b[i]=vw,vb
        
        return prev_w,prev_b       
   
    
            