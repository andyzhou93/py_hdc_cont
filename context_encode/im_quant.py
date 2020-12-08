import h5py
import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances, accuracy_score
from scipy.spatial.distance import cdist, pdist
from sklearn.model_selection import StratifiedKFold

import time
import pickle

def bipolarize(Y):
    X = np.copy(Y)
    X[X > 0] = 1.0
    X[X < 0] = -1.0
    X[X == 0] = np.random.choice([-1.0, 1.0], size=len(X[X == 0]))
    return X

def centroids(X,label=None):
    if label is not None:
        cLabel = np.unique(label)
        c = np.zeros((len(cLabel), X.shape[1]))
        for i,l in enumerate(cLabel):
            c[i,:] = bipolarize(np.sum(X[label==l],axis=0))
    else:
        c = bipolarize(np.sum(X,axis=0)).reshape(1,-1)
        cLabel = [0]
    return cLabel, c.astype('int')

def classify(v,am,metric):
    d = cdist(v,am,metric)
    label = np.argmin(d,axis=1)
    return label

dataFile = 'data.mat'

file = h5py.File(dataFile,'r')
experimentData = file['experimentData']
keys = list(experimentData.keys())

numTrials, numPositions, numGestures = experimentData[keys[0]].shape
D = file[experimentData[keys[4]][0,0,0]].shape[1]
numEx = 80
totalEx = numEx*numGestures*numPositions*numTrials

maxLevels = 64

resIM = {}

for N in range(maxLevels):
    print('\nRunning with %d quantization levels' % (N+1))
    
    start_t = time.time()

    ims = []
    for i in range(3):
        im = np.zeros((N+1,D))
        for x in range(N+1):
            im[x,:] = np.random.choice([-1.0, 1.0], size=D)
        ims.append(im)

    hv = np.zeros((totalEx,D),dtype=np.int8)
    gestLabel = np.zeros(totalEx,dtype=np.int8)
    posLabel = np.zeros(totalEx,dtype=np.int8)
        
    idx = np.arange(numEx).astype('int')
    for g in range(numGestures):
        for p in range(numPositions):
            for t in range(numTrials):
                expLabel = file[experimentData['expGestLabel'][t,p,g]][0,:]
                r = file[experimentData['emgHV'][t,p,g]][expLabel>0,:]
                accFeat = file[experimentData['accFeat'][t,p,g]][:,expLabel>0].T
                accIdx = np.round((accFeat + 1)/2*N).astype('int')
                accIdx[accIdx < 0] = 0
                accIdx[accIdx > N] = N
                xHV = ims[0][accIdx[:,0]]
                yHV = ims[1][accIdx[:,1]]
                zHV = ims[2][accIdx[:,2]]
                hv[idx,:] = xHV*yHV*zHV*r

                gestLabel[idx] = g
                posLabel[idx] = p
                idx += numEx

    # get list of unique values for each label
    gestures = np.unique(gestLabel)
    positions = np.unique(posLabel)

    combGP, groupGP = np.unique(np.column_stack((gestLabel,posLabel)),axis=0,return_inverse=True)

    numSplit = 10
    skf = StratifiedKFold(n_splits=numSplit)

    X = hv
    y = gestLabel
    c = posLabel
    g = groupGP

    accSingle = np.zeros((numPositions,numPositions,numSplit))
    accSeparate = np.zeros((numPositions,numSplit))
    accSuper = np.zeros((numPositions,numSplit))

    splitIdx = 0

    for trainIdx, testIdx in skf.split(X,g):
        print('Running iteration %d of %d...' % (splitIdx+1, numSplit))
        XTrain, XTest = X[trainIdx], X[testIdx]
        yTrain, yTest = y[trainIdx], y[testIdx]
        cTrain, cTest = c[trainIdx], c[testIdx]
        gTrain, gTest = g[trainIdx], g[testIdx]

        # generate a separate prototype for each arm position
        pLabel, p = centroids(XTrain,label=gTrain)

        # classify with each arm position only
        for pos in positions:
            AM = np.vstack([x for i,x in enumerate(p) if combGP[pLabel[i]][1] == pos])
            pred = classify(XTest,AM,'hamming')
            for posTest in positions:
                accSingle[pos,posTest,splitIdx] = accuracy_score(pred[cTest == posTest], yTest[cTest == posTest])

        # classify with all arm positions, separated
        pred = classify(XTest,p,'hamming')
        for posTest in positions:
            predLabel = [combGP[pr][0] for pr in pred[cTest == posTest]]
            accSeparate[posTest,splitIdx] = accuracy_score(predLabel,yTest[cTest == posTest])

        # classify with superimposed arm positions
        pLabel, p = centroids(XTrain,label=yTrain)
        pred = classify(XTest,p,'hamming')
        for posTest in positions:
            predLabel = [pLabel[pr] for pr in pred[cTest == posTest]]
            accSuper[posTest,splitIdx] = accuracy_score(predLabel,yTest[cTest == posTest])

        splitIdx += 1

    accSingle = np.mean(accSingle,axis=2)
    accSeparate = np.mean(accSeparate,axis=1)
    accSuper = np.mean(accSuper,axis=1)
    
    resIM[N+1] = (accSingle, accSeparate, accSuper)
    
    print('Took %f seconds' % (time.time() - start_t))

with open('resIM.pickle', 'wb') as f:
    pickle.dump(resIM, f, protocol=pickle.HIGHEST_PROTOCOL)