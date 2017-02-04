import numpy as np

from utils import dataLoaderMat, dataLoaderVec, slicer
from crossValidation import chooseBestSplitLen
from sklearn import model_selection
from sklearn import preprocessing

 #-------------------------------------------------
 #--------- LINEAR REGRESSION ---------------------
 #-------------------------------------------------

data, N = dataLoaderVec('tram')


from sklearn import linear_model as lm

# choose optimal split size (model dimension)

linReg = lm.LinearRegression()
tram_opt_dim = chooseBestSplitLen(data, linReg, max_size = 10, plot = True)
samples = slicer(data, tram_opt_dim, True)
X,Y = samples[:, :-1], samples[:,-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=0.66)
scaler = preprocessing.StandardScaler().fit(X)
Xtr = scaler.transform(Xtr)
Xva = scaler.transform(Xva) 
linReg.fit(Xtr, Ytr)
print('optimal split size' , tram_opt_dim)
print('Training error linReg Trams: '  , 1-linReg.score(Xtr,Ytr))
print('Validation error linReg Trams: ', 1-linReg.score(Xva,Yva))


# same as above, using cars

data, N = dataLoaderVec('car')
linReg = lm.LinearRegression()
car_opt_dim = chooseBestSplitLen(data, linReg, max_size = 10, plot = True)
samples = slicer(data, car_opt_dim, True)
X,Y = samples[:, :-1], samples[:,-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=0.66)
scaler = preprocessing.StandardScaler().fit(X)
Xtr = scaler.transform(Xtr)
Xva = scaler.transform(Xva) 

linReg.fit(Xtr, Ytr)
print('optimal split size' , car_opt_dim)
print('Training error linReg Cars: '  , 1-linReg.score(Xtr,Ytr))
print('Validation error linReg Cars: ', 1-linReg.score(Xva,Yva))


#-------------------------------------------
#------------ USING SVMs -------------------
#-------------------------------------------

print('\n------------------------------')
print('------- USING SVMs --------')
print('-------------------------------')

from sklearn import svm

#----- carica i dati ----- 
data, N = dataLoaderVec('tram')

testPerc = 0.66

# choose optimal split size (model dimension)

svr = svm.LinearSVR(C=0.03, epsilon=0.5)
tram_opt_dim = chooseBestSplitLen(data, svr, max_size = 10, plot = True)
samples = slicer(data, tram_opt_dim, True)
X,Y = samples[:, :-1], samples[:,-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=testPerc)
scaler = preprocessing.StandardScaler().fit(X)
Xtr = scaler.transform(Xtr)
Xva = scaler.transform(Xva) 

svr.fit(Xtr, Ytr)
print('optimal split size' , tram_opt_dim)
print('Training error linReg Trams: '  , 1-svr.score(Xtr,Ytr))
print('Validation error linReg Trams: ', 1-svr.score(Xva,Yva))


# choose optimal model parameters
tram_opt_dim = 5
data, N = dataLoaderVec('tram')
parameters = {'C':[0.01, 0.03, 0.05, 0.07, 0.09, 0.1], 'epsilon':[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
svr = svm.LinearSVR()
clf = model_selection.GridSearchCV(svr, parameters, verbose = 0)
samples = slicer(data, tram_opt_dim, True)
X,Y = samples[:, :-1], samples[:,-1]
print(X[0,:], Y[0])
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=testPerc)
scaler = preprocessing.StandardScaler().fit(X)
Xtr = scaler.transform(Xtr)
Xva = scaler.transform(Xva) 

print('Xva shape: ', Xva.shape)
print('Yva shape: ', Yva.shape)
clf.fit(Xtr, Ytr)
print(clf.best_params_)
print('Training error optimal SVM tram: '  , 1-clf.score(Xtr,Ytr))
print('Validation error optimal SVM tram: ', 1-clf.score(Xva,Yva))

# training with optimal parameters
svr = clf.best_estimator_
svr.fit(Xtr, Ytr)
Yva_hat = svr.predict(Xva)

from matplotlib import pyplot as plt

plt.figure()
plt.plot(Yva_hat[1000:1080], label='Predicted value')
plt.plot(Yva[1000:1080], label='Measured value')
plt.ylabel('KB/s')
plt.show()

# ----- FILTRO DI KALMAN -----

from kalman import kalmanFilter

best_params = clf.best_estimator_.coef_
print('tram_opt_dim: ', tram_opt_dim)
predictions = kalmanFilter(best_params,Xva)


plt.figure()
plt.plot(predictions[1000:1100])
plt.plot(Yva[1000:1100])
plt.show()
   