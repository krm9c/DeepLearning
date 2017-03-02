#############################################################################
# Pre Existing Libraries
from math import *
import os
import sys
import random
import time
import numpy as np
import itertools
import math
from collections import Counter
from numpy import genfromtxt
import heapq
from   sklearn.decomposition         import PCA
import matplotlib.pyplot                 as plt
from   matplotlib.colors             import ListedColormap
from   sklearn.cross_validation      import train_test_split
from   sklearn.preprocessing         import StandardScaler
from   sklearn.datasets              import make_moons, make_circles, make_classification
from   sklearn.neighbors             import KNeighborsClassifier
from   sklearn.svm                   import SVC
from   sklearn.tree                  import DecisionTreeClassifier
from   sklearn.ensemble              import RandomForestClassifier, AdaBoostClassifier
from   sklearn.naive_bayes           import GaussianNB
from   sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from   sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.random_projection 		 import SparseRandomProjection
from sklearn 						 import manifold
from   sklearn                       import cluster, datasets
from   sklearn.neighbors             import kneighbors_graph
from   sklearn.preprocessing         import StandardScaler
#######################################################################################
# Libraries created by us
# For windows use this path and comment the other
# sys.path.append('C:\Users\krm9c\Dropbox\Work\Research\Common_Libraries')
# path= "E:\Research_Krishnan\Data\Data-case-study-1"


# For MAC or Unix use this Path
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Common_Libraries')
# Set path for the data too
path = "/Users/krishnanraghavan/Documents/Data-case-study-1"


from Library_Paper_two       import *
from Library_Paper_one       import import_data, traditional_MTS


###############################################################################
def RollingDataImport():
	# Start_Analysis_Bearing()
	IR    	=  np.loadtxt('IR_sample.csv', delimiter=',')
	OR    	=  np.loadtxt('OR_sample.csv', delimiter=',')
	NL    	=  np.loadtxt('NL_sample.csv', delimiter=',')
	Norm  	=  np.loadtxt('Norm.csv'     , delimiter=',')

	sheet    = 'Test';
	f        = 'IR1.xls'
	filename =  os.path.join(path,f);
	Temp_IR  =  np.array(import_data(filename,sheet, 1));

	sheet    = 'Test';
	f        = 'OR1.xls'
	filename =  os.path.join(path,f);
	Temp_OR  =  np.array(import_data(filename,sheet, 1));

	sheet    = 'Test';
	f        = 'NL1.xls'
	filename =  os.path.join(path,f);
	Temp_NL  =  np.array(import_data(filename,sheet, 1));

	sheet    = 'normal';
	f        = 'Normal_1.xls'
	filename = os.path.join(path,f);
	Temp_Norm= np.array(import_data(filename,sheet, 1));


	return Temp_Norm, Temp_IR, Temp_OR, Temp_NL, IR, NL, OR, Norm

###############################################################################
def DeepLearn(T, Y, N_class):
	from keras.optimizers import SGD
	from keras.layers     import Input, Embedding, LSTM, Dense, merge, Activation
	from keras.models     import Sequential
	import numpy as np
	from keras.utils.np_utils import to_categorical
	from keras.layers import Merge
	labels = to_categorical(Y, N_class)
	X_train, X_test, Y_train, Y_test = train_test_split( X, labels,test_size=0.5, random_state=42)
	model = Sequential()
	model.add(Dense(1000, input_dim = T.shape[1],  activation='tanh'))
	model.add(Dense(500, activation='tanh'))
	model.add(Dense(250, activation='tanh'))
	model.add(Dense(125, activation='tanh'))
	model.add(Dense(64, activation='tanh'))
	model.add(Dense(32, activation='tanh'))
	model.add(Dense(16, activation='tanh'))
	model.add(Dense(8, activation='tanh'))
	model.add(Dense(N_class, activation='softmax'))
	model.compile(optimizer='rmsprop', loss='kld', metrics=['accuracy'])
	hist = model.fit(X_train, Y_train, nb_epoch=200, batch_size=100)
	Accuracy_arr = np.array(hist.history['acc']);
	Loss_arr = np.array(hist.history['loss']);
	np.savetxt('Loss_A.csv',  Loss_arr ,delimiter=',')
	np.savetxt('ACC_A.csv',   Accuracy_arr ,delimiter=',')
	model.save('Roll_myMod_A.h5')
	from keras.utils.np_utils import to_categorical
	loss, acc = model.evaluate(X, labels, verbose=0)
	print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
	Prob_array = np.array(model.predict_proba(X));
	np.savetxt('Pred_prob_A.csv', Prob_array, delimiter = ',')
	return model
def Artificial_Data(n_sam, n_fea, n_inf):
	X,y = make_classification(n_samples=n_sam, n_features=n_fea, n_informative=n_inf, n_redundant=(n_fea-n_inf), n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.01,class_sep=1.0, hypercube=True, shift=3.0, scale=1.0, shuffle=True, random_state= 9000)


	index_1 = [i for i,v in enumerate(y) if v == 0 ]

	index_2 = [i for i,v in enumerate(y) if v == 1 ]

	Data_class_1 = X[index_1,:]
	L1 = y[index_1];
	L2 = y[index_2];
	Data_class_2 = X[index_2,:]
	X = np.concatenate((Data_class_1, Data_class_2))
	Y = np.concatenate((np.zeros(Data_class_1.shape[0]),np.zeros(Data_class_2.shape[0])+1))
	return X, Y
def RollingElementBearing():
	Temp_Norm, Temp_IR, Temp_OR, Temp_NL, IR, NL, OR, Norm = RollingDataImport()
	X   =  np.concatenate((Norm, Temp_OR, Temp_IR, Temp_NL))
	Y    =  np.concatenate((np.zeros(Norm.shape[0]),np.zeros(Temp_OR.shape[0])+1, np.zeros(Temp_IR.shape[0])+2, np.zeros(Temp_NL.shape[0])+3))
	return X, Y;
def GasSensorArray():
	X,y = make_classification(n_samples=100000, n_features=432, n_informative=200, n_redundant=(232), n_classes=10, n_clusters_per_class=1, weights=None, flip_y=0.01,class_sep=1.0, hypercube=True, shift=3.0, scale=1.0, shuffle=True, random_state= 9000)


	index_1 = [i for i,v in enumerate(y) if v == 0 ]
	index_2 = [i for i,v in enumerate(y) if v == 1 ]
	index_3 = [i for i,v in enumerate(y) if v == 2 ]
	index_4 = [i for i,v in enumerate(y) if v == 3 ]
	index_5 = [i for i,v in enumerate(y) if v == 4 ]
	index_6 = [i for i,v in enumerate(y) if v == 5 ]
	index_7 = [i for i,v in enumerate(y) if v == 6 ]
	index_8 = [i for i,v in enumerate(y) if v == 7 ]
	index_9 = [i for i,v in enumerate(y) if v == 8 ]
	index_10 = [i for i,v in enumerate(y) if v == 9 ]

	Data_class_1 = X[index_1,:]
	Data_class_2 = X[index_2,:]
	Data_class_3 = X[index_3,:]
	Data_class_4 = X[index_4,:]
	Data_class_5 = X[index_5,:]
	Data_class_6 = X[index_6,:]
	Data_class_7 = X[index_7,:]
	Data_class_8 = X[index_8,:]
	Data_class_9 = X[index_9,:]
	Data_class_10 = X[index_10,:]

	X = np.concatenate((Data_class_1, Data_class_2,Data_class_3,Data_class_4,Data_class_5,Data_class_6,Data_class_7,Data_class_8,Data_class_9,Data_class_10))
	Y = np.concatenate(( np.zeros(Data_class_1.shape[0]), (np.zeros(Data_class_2.shape[0])+1), (np.zeros(Data_class_3.shape[0])+2), (np.zeros(Data_class_4.shape[0])+3), (np.zeros(Data_class_5.shape[0])+4), (np.zeros(Data_class_6.shape[0])+5), (np.zeros(Data_class_7.shape[0])+6), (np.zeros(Data_class_8.shape[0])+7), (np.zeros(Data_class_9.shape[0])+8), (np.zeros(Data_class_10.shape[0])+9)))
	return X, Y

if __name__ == "__main__":
    # Full Analysis Function Calls for analyzing Rolling element Bearing Data-set
	X, Y =  GasSensorArray()
	print X.shape
	print Y.shape
	X_scaler = StandardScaler()
	X_scaler.fit(X)
	T= X_scaler.transform(X)
	model = DeepLearn(T, Y, 10)
