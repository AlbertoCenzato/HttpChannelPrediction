import numpy as np
from sklearn import model_selection 
from matplotlib import pyplot as pl

from utils import slicer

def chooseBestSplitLen(data, model, max_size, testPerc = 0.33, plot = False):
   training_error   = np.zeros((max_size))
   validation_error = np.zeros((max_size))

   for sliceLen in range(2,max_size):
      X,Y = slicer(data,sliceLen, True)
      Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=testPerc)
      
      model.fit(Xtr, Ytr)
      training_error[sliceLen]   = 1-model.score(Xtr,Ytr)
      validation_error[sliceLen] = 1-model.score(Xva,Yva)
   
   if(plot):
      pl.figure()
      pl.plot(training_error, 'b', label='Training error')
      pl.plot(validation_error,'r', label='Validation error')
      pl.xlabel('Model dimension')
      pl.ylabel('Error')
      pl.grid()
      pl.legend(loc=4)
      pl.show()

   return np.argmin(validation_error[2:]) + 2