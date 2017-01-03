import numpy as np

from utils import dataLoaderMat, dataLoaderVec, slicer
from crossValidation import chooseBestSplitLen

 #-------------------------------------------------
 #--------- LINEAR REGRESSION ---------------------
 #-------------------------------------------------

data, N = dataLoaderVec('tram')

from sklearn import linear_model as lm

linReg = lm.LinearRegression()
len = chooseBestSplitLen(data, linReg, max_size = 30, plot = True)

'''
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
'''

# same as above, using cars

data, N = dataLoaderVec('car')

linReg = lm.LinearRegression()
len = chooseBestSplitLen(data, linReg, max_size = 30, plot = True)

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