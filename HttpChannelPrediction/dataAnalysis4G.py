import numpy as np

from utils import dataLoaderMat, dataLoaderVec, slicer
from crossValidation import chooseBestSplitLen
from sklearn import model_selection

 #-------------------------------------------------
 #--------- LINEAR REGRESSION ---------------------
 #-------------------------------------------------

data, N = dataLoaderVec('tram')

from sklearn import linear_model as lm

linReg = lm.LinearRegression()
tram_opt_dim = chooseBestSplitLen(data, linReg, max_size = 10, plot = True)
samples = slicer(data, tram_opt_dim, True)
X,Y = samples[:, :tram_opt_dim-1], samples[:,tram_opt_dim-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=0.66)
linReg.fit(Xtr, Ytr)
print('optimal split size' , tram_opt_dim)
print('Training error linReg Trams: '  , 1-linReg.score(Xtr,Ytr))
print('Validation error linReg Trams: ', 1-linReg.score(Xva,Yva))


# same as above, using cars

data, N = dataLoaderVec('car')
linReg = lm.LinearRegression()
car_opt_dim = chooseBestSplitLen(data, linReg, max_size = 10, plot = True)
samples = slicer(data, car_opt_dim, True)
X,Y = samples[:, :car_opt_dim-1], samples[:,car_opt_dim-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=0.66)
linReg.fit(Xtr, Ytr)
print('optimal split size' , car_opt_dim)
print('Training error linReg Cars: '  , 1-linReg.score(Xtr,Ytr))
print('Validation error linReg Cars: ', 1-linReg.score(Xva,Yva))


#-------------------------------------------
#------------ USING SVMs -------------------
#-------------------------------------------

#----- carica i dati ----- 
data, N = dataLoaderVec('tram')

#----- training -----
testPerc = 0.66

from sklearn import svm

data, N = dataLoaderVec('tram')
parameters = {'C':[0.001, 0.01, 0.1, 100000, 1000000], 'epsilon':[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
svr = svm.LinearSVR()
clf = model_selection.GridSearchCV(svr, parameters, verbose = 3)
samples = slicer(data, tram_opt_dim, True)
X,Y = samples[:, :tram_opt_dim-1], samples[:,tram_opt_dim-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=testPerc)
print('Xva shape: ', Xva.shape)
print('Yva shape: ', Yva.shape)
clf.fit(Xtr, Ytr)
print(clf.best_params_)
print('Training error optimal SVM tram: '  , 1-clf.score(Xtr,Ytr))
print('Validation error optimal SVM tram: ', 1-clf.score(Xva,Yva))


# ----- FILTRO DI KALMAN -----

from kalman import kalmanFilter

best_params = clf.best_estimator_.coef_
predictions = kalmanFilter(best_params,Xva)
for t in range(0,car_opt_dim(predictions)):
   print('predizione: ', predictions[t])
   print('valore vero: ', Yva[t])

from matplotlib import pyplot as plt

plt.plot(predictions)
plt.show()
   