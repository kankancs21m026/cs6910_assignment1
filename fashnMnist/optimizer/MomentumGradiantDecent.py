import numpy as np
import sys
from fashnMnist.NeuralNetwork import NeuralNetwork
class MomentumGradiantDecent(NeuralNetwork):
    def __init__(self, x, y, lr = .5,  wandb=None,wandbLog=False,x_val=None,y_val=None,epochs =100,batch=500,HiddenLayerNuron=[60,10],activation='tanh',decay_rate=0.01,initializer='he',weight_decay=0,dropout_rate=0,runlog=True,lossfunction='cross'
                ):
          
                # invoking the __init__ of the parent class 
                NeuralNetwork.__init__(self, x, y,wandb=wandb,wandbLog=wandbLog,x_val=x_val,y_val=y_val, lr = lr,dropout_rate=dropout_rate,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,decay_rate=decay_rate,initializer=initializer,weight_decay=weight_decay,runlog=runlog,lossfunction=lossfunction )
                
                
          
    def train(self):
        #initialize all parameters
        if(self.runlog):
            print('Starting Momentum Gradient Descent')
            print('.....................................')
        v_w, v_b  = self.DW, self.DB
        prevacc=0
        prevloss=999999
        for epoch in range(self.epochs):
            #reset weight derivatives and shuffle data
            self.resetWeightDerivative()
            self.shuffle()
            #control momentum
            gamma=self.momentumUpdate(epoch+1)
            
            for i in range(0, self.x.shape[0], self.batch):
                self.xBatch =self.x[i:i+self.batch]
                self.yBatch  = self.y[i:i+self.batch]
                pred=self.feedforward()
                self.backprop()
                prevW=self.W
                prevB=self.b
                prevvw=v_w
                prevvb=v_b    
                #Update parameter and return new v_w and v_b
                v_w, v_b=self.updateParam(v_w, v_b,epoch)
                   
            #verify loss after each epoch
            self.xBatch = self.x
            self.yBatch  =self.y
            pred=self.feedforward()
            acc=self.accurecy(pred,self.y)
            loss=self.calculateLoss()
            if(loss>prevloss):
                self.lr= self.stepDecay(epoch)
                self.W=prevW
                self.b=prevB
                acc=prevacc
                v_w=prevvw
                v_b=prevvb
                loss=prevloss
            else:
                prevacc =acc
                prevloss=loss
            
            #print details 
            self.printDetails(epoch,self.epochs,acc,loss)
            self.runAccurecy.append(acc)
            self.runLoss.append(loss)
        if(self.runlog):
            print()
            print('Completed')
            print('.....................................')
        
    def updateParam(self,v_w,v_b,epoch): 
        totalLayer=len(self.HiddenLayerNuron)
        gamma=self.getGamma(epoch)
        for i in range(totalLayer):
            v_w[i]= gamma*v_w[i]+(self.lr)* self.DW[i]  
            v_b[i]= gamma*v_b[i]+(self.lr)* self.DB[i]
            self.W[i] = self.W[i] - (self.lr)*v_w[i]
            self.b[i] = self.b[i] - (self.lr)* v_b[i]
            
        return v_w,v_b
   
    def getGamma(self,epoch):
        x=np.log((epoch/250)+1)
        x=-1-1*(x)
        x=2**x
        x=1-x
        return min(x,.9)
            