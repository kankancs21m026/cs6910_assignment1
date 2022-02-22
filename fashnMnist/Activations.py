import numpy as np
import math
class Activations:
    def __init__(self, activation='tanh'):
        self.act=activation
    
    def applyActivation(self,x,act=''):
        if(act==''):
            act=self.act
        if(act=='tanh'):
            return self.tanh(x)
        elif(act=='sigmoid'):
            return self.sig(x)
        elif(act=='relu'):
            return self.reLU(x)
        elif(act=='softmax'):
            return self.softmax(x)
        else:
             return x
            
            
            
    def applyActivationDeriv(self,x,act=''):
        if(act==''):
            act=self.act
        if(act=='tanh'):
            return self.dtanh(x)
        elif(act=='sigmoid'):
            return self.dsig(x)
        elif(act=='relu'):
            return self.dReLU(x)
       
        else:
             return x
        
    def sig(self,x):
          return  1/(1+np.exp(-x))

    # Sigmoidal derivative
    def dsig(self,x):
          return self.sig(x) * (1- self.sig(x))
        
        
    def reLU(self, x):
        #return np.maximum(0,x)
        return  np.where(x < 0, 0, x)
    
    def dReLU(self,x):
        return 1 * (x > 0) 

    
    
    
    def tanh(self, x):
        return np.tanh(x)
    
    def dtanh(self,x):
        tanh_x = self.tanh(x)
        return (1 - np.square(tanh_x))
    
    
    
    
    def softmax(self, z): 
        
        z=np.exp(z)
        tmp=np.sum(z, axis = 1) 
        for i in range(z.shape[0]):       
            z[i]=z[i]/tmp[i]
        return z
    
           
         
   
