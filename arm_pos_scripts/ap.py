import h5py
import hdc_clust_labels as hdc
import numpy as np
from sklearn import svm
from itertools import combinations
import scipy.io as sio

import sys
imType = sys.argv[1] # randIM circIM
emgHVType = sys.argv[2] # emgHV emgHV64 emgHVCAR emgHVCARNorm emgHVCARRel emgHVCARZeroed emgHVNorm emgHVRel emgHVZeroed
contextType = sys.argv[3] # none random accel
clusterType = sys.argv[4] # single separate auto
autoThreshold = int(sys.argv[5]) # 0 - 100
numTrainPositions = int(sys.argv[6]) # 1 2 3 4 5
numIters = int(sys.argv[7])

if clusterType == 'auto':
    matName = emgHVType + '_' + contextType + '_' + clusterType + '_' + str(autoThreshold) + '_' + str(numTrainPositions) + '_' + str(numIters) + '.mat'
else:
    matName = emgHVType + '_' + contextType + '_' + clusterType + '_' + str(numTrainPositions) + '_' + str(numIters) + '.mat'

autoThreshold = float(autoThreshold/100)

dataFile = '/Users/andy/Research/py_hdc_cont/emg_mat/armPosition/' + imType + '_hv.mat'
# dataFile = '/global/home/users/andyz/py_hdc_cont/emg_mat/armPosition/' + imType + '_hv.mat'

# file is saved in hdf5 format
file = h5py.File(dataFile,'r')
experimentData = file['experimentData']
keys = list(experimentData.keys())

numTrials, numPositions, numGestures = experimentData[keys[0]].shape
D = file[experimentData[keys[4]][0,0,0]].shape[1]

if contextType == 'random':
    contextVec = np.random.choice([-1.0, 1.0], size=(numPositions,D))

trainCombinations = list(combinations(np.arange(numPositions),numTrainPositions))
numCombinations = len(trainCombinations)

# output data to be put into struct
meanHDAcc = np.zeros((numCombinations,numPositions))

# keep track of clustering only if separate clustering is used
if clusterType == 'separate' or clusterType == 'auto':
    clustHits = np.zeros((numCombinations,numPositions,numPositions))
    clustCorrectHits = np.zeros((numCombinations,numPositions,numPositions))
    clustIncorrectHits = np.zeros((numCombinations,numPositions,numPositions))

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
            for apTrain in trainCombinations[apComb]:
                for g in range(numGestures):
                    for t in range(numTrials):
                        if t == trainTrials[g]:
                            expLabel = file[experimentData['expGestLabel'][t,apTrain,g]][0,:]
                            ng = file[experimentData[emgHVType][t,apTrain,g]][expLabel>0,:]
                            
                            if contextType == 'random':
                                ng = ng*contextVec[apTrain]
                            elif contextType == 'accel':
                                accHV = file[experimentData['accHV64'][t,apTrain,g]][expLabel>0,:]
                                ng = ng*accHV
                            
                            if clusterType == 'single':
                                AM.train(ng,vClass=g,vClust=0)
                            elif clusterType == 'separate':
                                AM.train(ng,vClass=g,vClust=apTrain)
                            elif clusterType == 'auto':
                                AM.train_sub_cluster(ng,vClass=g,vClustLabel=apTrain,threshold=autoThreshold)
            
            if clusterType == 'auto':
                AM.prune(min=numTrainPositions)
                
            # test AM
            for g in range(numGestures):
                for t in range(numTrials):
                    if t != trainTrials[g]:
                        expLabel = file[experimentData['expGestLabel'][t,apTest,g]][0,:]
                        ng = file[experimentData[emgHVType][t,apTest,g]][expLabel>0,:]
                        
                        if contextType == 'random':
                            ng = ng*contextVec[apTest]
                        elif contextType == 'accel':
                            accHV = file[experimentData['accHV64'][t,apTest,g]][expLabel>0,:]
                            ng = ng*accHV
                        
                        label,clust,clustLabel,sim = AM.match(np.asarray(ng),bipolar=True)
                        clust = np.asarray(clustLabel)
                        label = np.asarray(label)
                        hdAcc.append(np.sum(label == g)/len(label))
                        
                        if clusterType == 'separate' or clusterType == 'auto':
                            for p in range(numPositions):
                                clustHits[apComb,apTest,p] += sum(clust == p)
                                clustCorrectHits[apComb,apTest,p] += sum(clust[label==g] == p)
                                clustIncorrectHits[apComb,apTest,p] += sum(clust[label!=g] == p)
                                
        meanHDAcc[apComb,apTest] = np.mean(hdAcc)

matOut = {}
matOut['meanHDAcc'] = meanHDAcc
matOut['trainCombinations'] = trainCombinations
if clusterType == 'separate' or clusterType == 'auto':
    matOut['clustHits'] = clustHits
    matOut['clustCorrectHits'] = clustCorrectHits
    matOut['clustIncorrectHits'] = clustIncorrectHits
    
sio.savemat('./outputs/' + matName, matOut)