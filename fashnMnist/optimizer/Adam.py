import numpy as np
import sys
from fashnMnist.NeuralNetwork import NeuralNetwork
class Adam(NeuralNetwork):
    def __init__(self, x, y, lr = .5,wandb=None,wandbLog=False, x_val=None,y_val=None, epochs =100,batch=32,HiddenLayerNuron=[32,10],activation='tanh',beta1=0.9,beta2=0.99,decay_rate=0,initializer='he',dropout_rate=0,weight_decay=0,runlog=True,lossfunction='cross'):
          
                # invoking the __init__ of the parent class 
                NeuralNetwork.__init__(self, x, y, lr = lr,  wandb=wandb,wandbLog=wandbLog, x_val=x_val,y_val=y_val, epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,beta1=beta1,beta2=beta2,decay_rate=decay_rate,initializer=initializer,dropout_rate=dropout_rate,runlog=runlog,lossfunction=lossfunction)
                
                
          
    def train(self):
        #initialize all parameters
        if(self.runlog):
            print('Starting Adam')
            print('.....................................')
        m_w,v_w,m_b, v_b  = self.DW, self.DW, self.DB, self.DB
        prevacc=0
        prevloss=999999
        step=1
        beta1=.9
        beta2=0.9
        chunkSize=self.xBatch.shape[0]
        chunkSize=max(int(chunkSize/100),1)
        processedDataSize=0
        for epoch in range(self.epochs):
           
            self.shuffle()
           
            #don't want Stochastic adam as it takes crazy amount of time
            for i in range(0, self.x.shape[0], self.batch):
                self.resetWeightDerivative()
                #using mimentum update as it gives good accurecy instead any constant value
                beta1=self.momentumUpdate(step)
                beta2=self.momentumUpdate(step)
                self.xBatch =self.x[i:i+self.batch]
                self.yBatch  = self.y[i:i+self.batch]
                
                pred=self.feedforward()
               
                self.backprop()
                
                step+=1
                processedDataSize=processedDataSize+self.batch 
                m_w,v_w,m_b, v_b =self.updateParam( m_w,v_w,m_b, v_b ,beta1,beta2,(step))
               
            
            #verify loss after each epoch
            self.xBatch = self.x
            self.yBatch  =self.y
            pred=self.feedforward()
            acc=self.accurecy(pred,self.y)
            step+=1
            loss=self.calculateLoss() 
            processedDataSize=0

            #print details       
            self.printDetails(epoch,self.epochs,acc,loss)
            self.runAccurecy.append(acc)
            self.runLoss.append(loss)
        print()
        if(self.runlog):
            print('Completed')
            print('.....................................')
   
        
    def updateParam(self, m_w,v_w,m_b, v_b ,beta1,beta2,epoch): 
        totalLayer=len(self.HiddenLayerNuron)
        
        beta1Hat=1.0-(beta1**epoch)
        beta2Hat=1.0-(beta2**epoch)
        
        beta1Dash=(1.0-beta1)
        beta2Dash=(1.0-beta2)
        eps=.00001#small number
        
        newvw=[]
        newvb=[]
        newmw=[]
        newmb=[]
        for i in range(totalLayer):
            vw1= np.multiply(v_w[i],beta2)
            vw2=np.square(self.DW[i])
            vw2= np.multiply(vw2 ,beta2Dash)
            vw3=np.add(vw1,vw2)
           
            vb1=np.multiply(v_b[i],beta2)
            vb2=np.square(self.DB[i])
            vb2=np.multiply(vb2 ,beta2Dash)
            
            vb3=np.add(vb1,vb2)
           

            mw1= np.multiply(m_w[i],beta1)
            mw2= np.multiply(self.DW[i] ,beta1Dash)
            mw=np.add(mw1,mw2)
            mb1= np.multiply(m_b[i],beta1)
            mb2= np.multiply(self.DB[i] ,beta1Dash)
            mb=np.add(mb1,mb2)

             #bias correction
            vw= vw3/beta2Hat
            vb= vb3/beta2Hat 
            mw= mw/beta1Hat
            mb= mb/beta1Hat
            
            newvw.append(vw)
            newvb.append(vb)
            newmw.append(mw)
            newmb.append(mb)
            
            vw= np.sqrt(vw)+eps
            vb= np.sqrt(vb)+eps
            
           
            self.W[i] = self.W[i] - (self.lr/vw)*(mw)
            self.b[i] = self.b[i] - (self.lr/vb)* (mb)
         
        return newmw,newvw,newmb,newvb
   
    
            