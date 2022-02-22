import numpy as np
from keras.datasets import fashion_mnist

class Preprocessor:
    def __init__(self,normalization=False):
        self.mean=[]
        self.sd=[]
        self.normalization=normalization
        
   
    def Process_Fashon_mnistDataSet(self,x_train,y_train,x_test,y_test):  
        x_train=self.normalize(x_train)
        x_test=self.normalize(x_test)
        x_train=x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
        x_test=x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
        y_train=self.oneHotEncoding(y_train) 
        y_test=self.oneHotEncoding(y_test) 
        if(self.normalization):
            x_train=self.fitStandarization(x_train)
            x_test=self.applyStandarization(x_test)
        return x_train,y_train,x_test,y_test
        
    def oneHotEncoding(self,y_train):
        datasetTrain=y_train.shape[0]
        classifier,count = np.unique(y_train, return_counts=True)
        y_train_classifier=np.zeros((datasetTrain,10))
        rows=0
        for i in  y_train:
            y_train_classifier[rows][i.astype('int')]=1 
            rows+=1;
        return y_train_classifier

    def normalize(self,x):
        return x.astype("float64") /255       
            
    def fitStandarization(self,x):
        self.mean=np.mean(x,axis=0)
        var=np.var(x,axis=0)
        self.sd=np.sqrt(var)
        for i in range(x.shape[1]):
            x[:,i]= x[:,i]-self.mean[i]
            x[:,i]=x[:,i]/self.sd[i]
        return x
     
    def applyStandarization(self,x):
        
        if(len(self.mean)==0):
            self.mean=np.mean(x,axis=0)
            var=np.var(x,axis=0)
            self.sd=np.sqrt(var)
        for i in range(x.shape[1]):
            x[:,i]= x[:,i]-self.mean[i]
            x[:,i]=x[:,i]/self.sd[i]    
        return x
        
    