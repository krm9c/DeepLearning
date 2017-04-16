#############################################################################
# Pre Existing Libraries
from math import *
import os
import tensorflow as tf
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

if __name__ == "__main__":
	# DataSet
	print "---MNIST---"
	### Import all the libraries required by us
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets(myRespath+"MNIST_data/",one_hot=True)
	model = DeepLearn(T, Y, 10)
