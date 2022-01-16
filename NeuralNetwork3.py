import numpy as np
import math
import csv
def cost(A67,label):
    sampleno=label.shape[1]
    print(sampleno)
    
    
    cost=-(1./sampleno)*(np.sum(label*np.log(A67)+(1-label)*np.log(1-A67)))
    cost=np.squeeze(cost)
    print(cost)
    
def back_softmax(x,out):
    Sz = softmax(x)
    
    
    N = x.shape[0]
    D = np.zeros((N, x.shape[1]))
    for i in range(N):
        for j in range(x.shape[1]):
            D[i, j] = Sz[i, 0] * (np.float32(i == j) - Sz[0, j])
    
    return D
    
    
    

def back_sigmoid(x,out):
    sig=1/(1+np.exp(-x))
    back=out*sig*(1-sig)
    return back

def sigmoid(x):
    sig=1/(1+np.exp(-x))
    
    return sig

def softmax(x):
    X=x.T
    
    max = np.max(X,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(X - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    soft = e_x / sum 
    
    
    return soft.T


def initialize(ldim):
    np.random.seed(2)
    parameters={}
    length=len(ldim)
    for i in range (1,length):
        
        parameters['w'+str(i)]=np.random.randn(ldim[i],ldim[i-1])*0.01
        parameters['b'+str(i)]=np.zeros((ldim[i],1))
    return parameters
        
def forwardprop(x,parameters):
    cache={}
    O1=np.dot(parameters['w1'],x)+parameters['b1']
    
    A1=sigmoid(O1)
    O2=np.dot(parameters['w2'],A1)+parameters['b2']
    A2=sigmoid(O2)
    O3=np.dot(parameters['w3'],A2)+parameters['b3']
    A3=softmax(O3)
    cache['X']=x
    cache['O1']=O1
    cache['A1']=A1
    cache['O2']=O2
    cache['A2']=A2
    cache['O3']=O3
    cache['A3']=A3
    
    return cache
def backprop(cache,parameters,label):
    grads={}
    dcost=cache['A3']-label
    dA3 = - np.divide(label, cache['A3']) +np.divide(1 - label, 1 - cache['A3'])
    
    #dO3=back_softmax(cache['O3'],dA3)
    
    m3=cache['A2'].shape[1]
    dw3=1./m3*np.dot(dcost,cache['A2'].T)
    db3=1./m3*np.sum(dcost,axis=1,keepdims=True)
    
    dA2=np.dot(parameters['w3'].T,dcost)
    dO2=back_sigmoid(cache['O2'],dA2)
    m2=cache['A1'].shape[1]
    dw2=1./m2*np.dot(dO2,cache['A1'].T)
    db2=1./m2*np.sum(dO2,axis=1,keepdims=True)
    
    dA1=np.dot(parameters['w2'].T,dO2)
    dO1=back_sigmoid(cache['O1'],dA1)
    m1=cache['X'].shape[1]
    dw1=1./m1*np.dot(dO1,cache['X'].T)
    db1=1./m1*np.sum(dO1,axis=1,keepdims=True)
    
    grads['dw3']=dw3
    grads['db3']=db3
    grads['dw2']=dw2
    grads['db2']=db2
    grads['dw1']=dw1
    grads['db1']=db1
    return grads
def update(parameters,caches,learningrate):
    
    parameters['w1']=parameters['w1']-learningrate*caches['dw1']
    parameters['b1']=parameters['b1']-learningrate*caches['db1']
    parameters['w2']=parameters['w2']-learningrate*caches['dw2']
    parameters['b2']=parameters['b2']-learningrate*caches['db2']
    parameters['w3']=parameters['w3']-learningrate*caches['dw3']
    parameters['b3']=parameters['b3']-learningrate*caches['db3']
    return parameters
def output(cache):
    l=cache['A3'].T
    f=open("test_predictions.csv","w")
    for i in range (l.shape[0]):
        max=-np.inf
        for j in range(l.shape[1]):
            if l[i][j]>max:
                max=l[i][j]
                predict=j
        f.write(str(predict)+"\n")
        
        
def model (X,Y,learning_rate):
    parameter=initialize([784,30,20,10])
    for i in range(20000):
        cache=forwardprop(X.T,parameter)
        if i%100==0:
            cost(cache['A3'],Y.T)
            #output(cache)
        updates=backprop(cache,parameter,Y.T)
        parameter=update(parameter,updates,learning_rate)
    return parameter
def predict(X,parameter):
    cache=forwardprop(X.T,parameter)
    output(cache)
    
    
trainimg = []
trainlabel=[]
with open("train_image.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        trainimg.append(row)
with open("train_label.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        trainlabel.append((np.array(row)))
  
encodingtrainlabel=np.zeros((len(trainimg),10))
i=0

for k in trainlabel:
    
    encodingtrainlabel[i][int(k[0])]=1
    i+=1
print(encodingtrainlabel.shape)

trainimg=np.array(trainimg).reshape(len(trainimg),784)/255

para=model(trainimg,encodingtrainlabel,0.75)
trainimg = []
with open("test_image.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        trainimg.append(row)
trainimg=np.array(trainimg).reshape(len(trainimg),784)/255
predict(trainimg,para)