import h5py
import hdc
import numpy as np
from sklearn import svm
from itertools import combinations
import scipy.io as sio

import sys
numTrainPositions = int(sys.argv[1])
numIters = 30
matName = 'relative_accelerometer_single_' + sys.argv[1]
cimLevels = sys.argv[2]

# dataFile = '/Users/andy/Research/py_hdc_cont/emg_mat/armPosition/sub1exp0.mat'
dataFile = '/global/home/users/andyz/py_hdc_cont/emg_mat/armPosition/sub1exp0_emgCIM.mat'

# file is saved in hdf5 format
file = h5py.File(dataFile,'r')
experimentData = file['experimentData']
keys = list(experimentData.keys())
    
numTrials, numPositions, numGestures = experimentData[keys[0]].shape
D = file[experimentData[keys[4]][0,0,0]].shape[1]

trainCombinations = list(combinations(np.arange(numPositions),numTrainPositions))
numCombinations = len(trainCombinations)

meanHDAcc = np.zeros((numCombinations,numPositions))
# meanSVMAcc = np.zeros((numCombinations,numPositions))

# clustHits = np.zeros((numCombinations,numPositions,numPositions))
# clustCorrectHits = np.zeros((numCombinations,numPositions,numPositions))
# clustIncorrectHits = np.zeros((numCombinations,numPositions,numPositions))

for apComb in range(numCombinations):
    for apTest in range(numPositions):
        hdAcc = []
        svmAcc = []
        for n in range(numIters):
            # set up new associative memory
            AM = []
            AM = hdc.Memory(D)
            # train/test split with single trial for training, remaining trials for testing
            trainTrials = np.random.randint(numTrials,size=numGestures)
            # train AM and build training array for SVM
            Xtrain = np.empty((0,320))
            ytrain = np.empty(0)
            for apTrain in trainCombinations[apComb]:
                for g in range(numGestures):
                    for t in range(numTrials):
                        if t == trainTrials[g]:
                            expLabel = file[experimentData['expGestLabel'][t,apTrain,g]][0,:]
                            ng = file[experimentData['emgHV'][t,apTrain,g]][expLabel>0,:]
                            accHV = file[experimentData['accHV'+cimLevels][t,apTrain,g]][expLabel>0,:]
                            ng = ng*accHV
                            AM.train(ng,vClass=g,vClust=0)
                            # AM.train(ng,vClass=g,vClust=apTrain)
#                             AM.train_sub_cluster(ng,vClass=g)
#                             AM.prune(min=5)

                            # # gather features for SVM (or other model)
                            # feat = file[experimentData['emgFeat'][t,apTrain,g]][:,expLabel>0].T
                            # numEx, numCh = feat.shape
                            # ngramLen = 5
                            # x = np.zeros((numEx-ngramLen+1,numCh*ngramLen))
                            # for i in range(ngramLen):
                            #     x[:,np.arange(numCh)+i*numCh] = feat[np.arange(numEx-ngramLen+1)+i,:]*6400
                            # Xtrain = np.concatenate((Xtrain,x))
                            # ytrain = np.concatenate((ytrain,g*np.ones(numEx-ngramLen+1)))
            
            # # train SVM (or other model)
            # clf = svm.SVC(decision_function_shape='ovo',kernel='linear',C=1)
            # clf.fit(Xtrain,ytrain)
            
            # test AM
            for g in range(numGestures):
                for t in range(numTrials):
                    if t != trainTrials[g]:
                        expLabel = file[experimentData['expGestLabel'][t,apTest,g]][0,:]
                        ng = file[experimentData['emgHV'][t,apTest,g]][expLabel>0,:]
                        accHV = file[experimentData['accHV'+cimLevels][t,apTest,g]][expLabel>0,:]
                        ng = ng*accHV
                        label,clust,sim = AM.match(np.asarray(ng),bipolar=True)
                        clust = np.asarray(clust)
                        label = np.asarray(label)
                        hdAcc.append(np.sum(label == g)/len(label))
                        
                        # for p in range(numPositions):
                        #     clustHits[apComb,apTest,p] += sum(clust == p)
                        #     clustCorrectHits[apComb,apTest,p] += sum(clust[label==g] == p)
                        #     clustIncorrectHits[apComb,apTest,p] += sum(clust[label!=g] == p)
                        
                        # feat = file[experimentData['emgFeat'][t,apTest,g]][:,expLabel>0].T
                        # numEx, numCh = feat.shape
                        # ngramLen = 5
                        # x = np.zeros((numEx-ngramLen+1,numCh*ngramLen))
                        # for i in range(ngramLen):
                        #     x[:,np.arange(numCh)+i*numCh] = feat[np.arange(numEx-ngramLen+1)+i,:]*6400
                        
                        # yhat = clf.predict(x)
                        # svmAcc.append(np.sum(yhat == g)/len(yhat))
        
        meanHDAcc[apComb,apTest] = np.mean(hdAcc)
        # meanSVMAcc[apComb,apTest] = np.mean(svmAcc)

matOut = {}
matOut['meanHDAcc'] = meanHDAcc
# matOut['meanSVMAcc'] = meanSVMAcc
# matOut['clustHits'] = clustHits
# matOut['clustCorrectHits'] = clustCorrectHits
# matOut['clustIncorrectHits'] = clustIncorrectHits
matOut['trainCombinations'] = trainCombinations

sio.savemat('./outputs/' + matName,matOut)
