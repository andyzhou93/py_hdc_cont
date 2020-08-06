import h5py
import hdc
import numpy as np
from sklearn import svm
from itertools import combinations
import scipy.io as sio

import sys
dataName = sys.argv[1] # allHV.npz
emgHVType = sys.argv[2] # hvRel hvRelAcc hvAbs hvAbsAcc
contextType = sys.argv[3] # none random
clusterType = sys.argv[4] # single separate auto
autoThreshold = int(sys.argv[5]) # 0 - 100
numTrainPositions = int(sys.argv[6]) # 1 - 8
crossTrial = sys.argv[7] # cross within
doLVQ = sys.argv[8] # lvqOn lvqOff
numIters = int(sys.argv[9])

if clusterType == 'auto':
    matName = emgHVType + '_' + contextType + '_' + clusterType + '_' + str(autoThreshold) + '_' + str(numTrainPositions) + '_' + crossTrial + '_' + doLVQ + '_' + str(numIters) + '.mat'
else:0
    matName = emgHVType + '_' + contextType + '_' + clusterType + '_' + str(numTrainPositions) + '_' + crossTrial + '_' + doLVQ +  '_' + str(numIters) + '.mat'

autoThreshold = float(autoThreshold/100)


allHV = np.load(dataName)
hv = allHV[emgHVType]
gestLabel = allHV['gestLabel']
posLabel = allHV['posLabel']
trialLabel = allHV['trialLabel']

gestures = np.unique(gestLabel)
positions = np.unique(posLabel)
trials = np.unique(trialLabel)

numGestures = len(gestures)
numPositions = len(positions)
numTrials = len(trials)

D = hv.shape[1]
numHV = 80

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

if clusterType == 'separate':
    prototypeSims = np.zeros((numCombinations, numTrainPositions*numGestures, numTrainPositions*numGestures))

for apComb in range(numCombinations):
    for apTest in range(numPositions):
        hdAcc = []
        for n in range(numIters):
            if crossTrial == 'within':
                trainIdx = []
                testIdx = []
                offset = 0
                for g in range(numGestures):
                    for p in range(numPositions):
                        for t in range(numTrials):
                            x = np.arange(numHV) + offset
                            np.random.shuffle(x)
                            trainIdx.extend(x[:round(numHV/4)])
                            testIdx.extend(x[round(numHV/4):])
                            offset += numHV
            elif crossTrial == 'cross':
                trainIdx = []
                testIdx = []
                offset = 0
                for g in range(numGestures):
                    for p in range(numPositions):
                        trainTrial = np.random.choice(np.arange(numTrials),1)[0]
                        for t in range(numTrials):
                            x = np.arange(numHV) + offset
                            if t == trainTrial:
                                trainIdx.extend(x)
                            else:
                                testIdx.extend(x)
                            offset += numHV

            trainIdx.sort()
            testIdx.sort()

            hvTrain = hv[trainIdx,:]
            hvTest = hv[testIdx,:]
            gestTrain = gestLabel[trainIdx]
            posTrain = posLabel[trainIdx]
            trialTrain = trialLabel[trainIdx]
            gestTest = gestLabel[testIdx]
            posTest = posLabel[testIdx]
            trialTest = trialLabel[testIdx]

            # set up new associative memory
            AM = []
            AM = hdc.Memory(D)
            for apTrain in trainCombinations[apComb]:
                for g in range(numGestures):
                    for t in range(numTrials):
                        ng = hvTrain[(gestTrain==g) & (posTrain==apTrain) & (trialTrain==t),:]
                        if ng.size != 0:
                            if contextType == 'random':
                                ng = ng*contextVec[apTrain]
                            
                            if clusterType == 'single':
                                AM.train(ng,vClass=g,vClust=0)
                            elif clusterType == 'separate':
                                AM.train(ng,vClass=g,vClust=apTrain)
                            elif clusterType == 'auto':
                                AM.train_sub_cluster(ng,vClass=g,vClustLabel=apTrain,threshold=autoThreshold)

            if clusterType == 'auto':
                AM.prune(min=numTrainPositions)

            if clusterType == 'separate':
                prototypeSims[apComb,:,:] += AM.match(AM.vec)[-1]/numIters
                
            # test AM
            for g in range(numGestures):
                for t in range(numTrials):
                    ng = hvTest[(gestTest==g) & (posTest==apTest) & (trialTest==t),:]
                    if ng.size != 0:
                        if contextType == 'random':
                            ng = ng*contextVec[apTest]
                        
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
if clusterType == 'separate':
    matOut['prototypeSims'] = prototypeSims
    
sio.savemat('./testout/' + matName, matOut)
