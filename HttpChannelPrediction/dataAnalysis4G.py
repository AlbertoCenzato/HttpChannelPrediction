import numpy as np
import pandas as pd
import os

def dataLoaderMat(path):
    files = os.listdir(path)
    data = []
    for f in files:
        df = pd.read_csv(path + '\\' + f, sep = ' ', usecols = [4, 5])
        data.append(df.values[:,0]/df.values[:,1])
    data = np.array([[x for x in y] for y in data])
    return data, len(data)

def dataLoaderVec(path):
    files = os.listdir(path)
    data = np.zeros((0,2))
    for f in files:
        df = pd.read_csv(path + '\\' + f, sep = ' ', usecols = [4, 5])
        data = np.vstack((data, df.values))
    return data[:,0]/data[:,1], data.shape[0]

def slicer(data, sliceLen, overlap):
    N = data.shape[0]
    if (not overlap):
        samples = np.zeros((N//sliceLen,sliceLen))
        for i in range(0,N//sliceLen):
            samples[i,:] = data[i*sliceLen:(1+i)*sliceLen]
        return samples
    samples = np.zeros((N-sliceLen,sliceLen))
    for i in range(0,N-sliceLen):
        samples[i,:] = data[i:i+sliceLen]
    return samples


 #-------------------------------------------------
 #--------- LINEAR REGRESSION ---------------------
 #-------------------------------------------------

 data, N = dataLoaderVec('tram')
from sklearn.cross_validation import train_test_split
from sklearn import linear_model as lm
testPerc = 0.33

max_size = 30
training_error   = np.zeros((max_size))
validation_error = np.zeros((max_size))

for sliceLen in range(2,max_size):
    print sliceLen
    samples = slicer(data,sliceLen, True)
    X,Y = samples[:, :sliceLen-1], samples[:,sliceLen-1]
    print X[:3,:], Y[:3]
    Xtr, Xva, Ytr, Yva = train_test_split(X, Y, test_size=testPerc)
    linReg = lm.LinearRegression()
    linReg.fit(Xtr, Ytr)
    training_error[sliceLen]   = 1-linReg.score(Xtr,Ytr)
    validation_error[sliceLen] = 1-linReg.score(Xva,Yva)

import matplotlib.pyplot as pl
pl.figure()
pl.plot(training_error, 'b', label='Training error')
pl.plot(validation_error,'r', label='Validation error')
pl.xlabel('Split size')
pl.ylabel('Error')
pl.grid()
pl.legend(loc=4)
pl.show()


# Here we predict a completely new trajectory out of the old ones

data, N = dataLoaderMat('tram')

train_size = 0.66
Ntr = int(np.ceil(N*train_size))
train = np.array([])
validation = np.array([])
for i in range(0,Ntr):
    train = np.append(train,data[i])
for i in range(Ntr,N):
    validation = np.append(validation,data[i])

sliceLen = 50
samples_tr = slicer(train,sliceLen, True)
samples_va = slicer(validation, sliceLen, True)

Xtr,Ytr = samples_tr[:, :sliceLen-2],samples_tr[:,sliceLen-1]
Xva,Yva = samples_va[:, :sliceLen-2],samples_va[:,sliceLen-1]

from sklearn import linear_model as lm
linReg = lm.LinearRegression(normalize=True)
linReg.fit(Xtr, Ytr)
print 'Training error: ', 1-linReg.score(Xtr,Ytr)
print 'Validation error: ', 1-linReg.score(Xva,Yva)

import matplotlib.pyplot as pl
pl.figure()
pl.plot(Yva[:100], 'b')
pl.plot(linReg.predict(Xva)[:100],'r')
pl.show()

pl.figure()
pl.plot(validation[:2000], 'b')
pl.show()


# same as above, using cars

data, N = dataLoaderMat('car')

train_size = 0.66
Ntr = int(np.ceil(N*train_size))
train = np.array([])
validation = np.array([])
for i in range(0,Ntr):
    train = np.append(train,data[i])
for i in range(Ntr,N):
    validation = np.append(validation,data[i])

sliceLen = 10
samples_tr = slicer(train,sliceLen, False)
samples_va = slicer(validation, sliceLen, False)

Xtr,Ytr = samples_tr[:, :sliceLen-2],samples_tr[:,sliceLen-1]
Xva,Yva = samples_va[:, :sliceLen-2],samples_va[:,sliceLen-1]

from sklearn import linear_model as lm
linReg = lm.LinearRegression(normalize=True)
linReg.fit(Xtr, Ytr)
print 'Training error: ', 1-linReg.score(Xtr,Ytr)
print 'Validation error: ', 1-linReg.score(Xva,Yva)

import matplotlib.pyplot as pl
pl.figure()
pl.plot(Yva[:100], 'b')
pl.plot(linReg.predict(Xva)[:100],'r')
pl.show()

pl.figure()
pl.plot(validation[:2000], 'b')
pl.show()


#-------------------------------------------
#------------ USING SVMs -------------------
#-------------------------------------------

#----- carica i dati ----- 
data, N = dataLoaderMat('car')
train_size = 0.66
Ntr = int(np.ceil(N*train_size))
X = np.array([])
train = np.array([])
validation = np.array([])
for i in range(0,N):
    X = np.append(X,data[i])
    
#----- normalizzazione ------
from sklearn import preprocessing
X = preprocessing.scale(X)

for i in range(0,N):
    if(i < Ntr):
        train = np.append(train,data[i])
    else:
        validation = np.append(validation,data[i])

#----- slicing -----
sliceLen = 10
samples_tr = slicer(train,sliceLen, False)
samples_va = slicer(validation, sliceLen, False)

Xtr,Ytr = samples_tr[:, :sliceLen-2],samples_tr[:,sliceLen-1]
Xva,Yva = samples_va[:, :sliceLen-2],samples_va[:,sliceLen-1]


#----- training -----
from sklearn import svm
svr = svm.SVR()
svr.fit(Xtr, Ytr)
print 'Training error: ', 1-svr.score(Xtr,Ytr)
print 'Validation error: ', 1-svr.score(Xva,Yva)

import matplotlib.pyplot as pl
pl.figure()
pl.plot(Yva[:100], 'b')
pl.plot(svr.predict(Xva)[:100],'r')
pl.show()

pl.figure()
pl.plot(validation[:2000], 'b')
pl.show()