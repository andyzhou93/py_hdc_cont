import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold

from multiprocessing import Pool

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

def run_sample(vals):
    CIMRanges = vals[:3]
    NQuants = vals[3:]
    dataFile = 'data.mat'
    file = h5py.File(dataFile,'r')
    experimentData = file['experimentData']
    keys = list(experimentData.keys())
    numTrials, numPositions, numGestures = experimentData[keys[0]].shape
    D = file[experimentData[keys[3]][0,0,0]].shape[1]
    numEx = 80

    ims = []
    for i in range(3):
        im = np.zeros((NQuants[i],D))
        seed = np.random.choice([-1.0, 1.0], size=D)
        flipBits = np.random.permutation(D)
        numFlip = np.round(np.linspace(0,CIMRanges[i],NQuants[i])).astype('int')
        for x in range(NQuants[i]):
            im[x,:] = seed
            im[x,flipBits[:numFlip[x]]] *= -1
        ims.append(im)

    totalEx = numEx*numGestures*numPositions*numTrials

    hv = np.zeros((totalEx,D),dtype=np.int8)
    gestLabel = np.zeros(totalEx,dtype='int')
    posLabel = np.zeros(totalEx,dtype='int')

    idx = np.arange(numEx).astype('int')
#     for g in tqdm(range(numGestures)):
    for g in range(numGestures):
        for p in range(numPositions):
            for t in range(numTrials):
                expLabel = file[experimentData['expGestLabel'][t,p,g]][0,:]
                r = file[experimentData['emgHV'][t,p,g]][expLabel>0,:]
                accFeat = file[experimentData['accFeat'][t,p,g]][:,expLabel>0].T

                accIdx = np.zeros(accFeat.shape,dtype='int')
                for ax in range(3):
                    accIdx[:,ax] = np.floor((accFeat[:,ax] + 1)/2*(NQuants[ax])).astype('int')
                    accIdx[accIdx[:,ax] < 0,ax] = 0
                    accIdx[accIdx[:,ax] > NQuants[ax]-1,ax] = NQuants[ax]-1

                xHV = ims[0][accIdx[:,0]]
                yHV = ims[1][accIdx[:,1]]
                zHV = ims[2][accIdx[:,2]]

                hv[idx,:] = r*xHV*yHV*zHV

                gestLabel[idx] = g
                posLabel[idx] = p
                idx += numEx

    combGP, groupGP = np.unique(np.column_stack((gestLabel,posLabel)),axis=0,return_inverse=True)
    gestures = np.unique(gestLabel)
    positions = np.unique(posLabel)
    numGestures = len(gestures)
    numPositions = len(positions)

    numSplit = 5
    skf = StratifiedKFold(n_splits=numSplit)

    X = hv
    y = gestLabel
    c = posLabel
    g = groupGP

    accSingle = np.zeros((numPositions,numPositions,numSplit))
    accSuperProto = np.zeros((numPositions,numSplit))
    accSuper = np.zeros((numPositions,numSplit))

    splitIdx = 0

#     for trainIdx, testIdx in tqdm(skf.split(X,g)):
    for trainIdx, testIdx in skf.split(X,g):
    #     print('Running iteration %d of %d...' % (splitIdx+1, numSplit))
        XTrain, XTest = X[trainIdx], X[testIdx]
        yTrain, yTest = y[trainIdx], y[testIdx]
        cTrain, cTest = c[trainIdx], c[testIdx]
        gTrain, gTest = g[trainIdx], g[testIdx]

        # generate a separate prototype for each gesture and arm position
        pLabel, p = centroids(XTrain,label=gTrain)

        # classify with each arm position only
        for pos in positions:
            AM = np.vstack([x for i,x in enumerate(p) if combGP[pLabel[i]][1] == pos])
            pred = classify(XTest,AM,'cosine')
            for posTest in positions:
                accSingle[pos,posTest,splitIdx] = accuracy_score(pred[cTest == posTest], yTest[cTest == posTest])

        # classify with superimposed arm positions - superimposed prototypes
        pLabel, p = centroids(p,label=combGP[:,0])
        pred = classify(XTest,p,'cosine')
        for posTest in positions:
            predLabel = [pLabel[pr] for pr in pred[cTest == posTest]]
            accSuperProto[posTest,splitIdx] = accuracy_score(predLabel,yTest[cTest == posTest])

        # classify with superimposed arm positions - superimposed examples
        pLabel, p = centroids(XTrain,label=yTrain)
        pred = classify(XTest,p,'cosine')
        for posTest in positions:
            predLabel = [pLabel[pr] for pr in pred[cTest == posTest]]
            accSuper[posTest,splitIdx] = accuracy_score(predLabel,yTest[cTest == posTest])

        splitIdx += 1

    accSingle = np.mean(accSingle,axis=2)
    accSuperProto = np.mean(accSuperProto,axis=1)
    accSuper = np.mean(accSuper,axis=1)

#     accIn = np.zeros(numPositions)
#     accOut = np.zeros(numPositions)
#     accAll = np.zeros(numPositions)
#     for p in positions:
#         accIn[p] = np.mean(accSingle[p][positions == p])
#         accOut[p] = np.mean(accSingle[p][positions != p])
#         accAll[p] = np.mean(accSingle[p])

#     print('AccIn = %.2f%%' % (np.mean(accIn)*100), end=' ')
#     print(['%.2f%%' % d for d in accIn*100])

#     print('AccOut = %.2f%%' % (np.mean(accOut)*100), end=' ')
#     print(['%.2f%%' % d for d in accOut*100])

#     print('AccAll = %.2f%%' % (np.mean(accAll)*100), end=' ')
#     print(['%.2f%%' % d for d in accAll*100])

#     print('AccSuper = %.2f%%' % (np.mean(accSuper)*100), end=' ')
#     print(['%.2f%%' % d for d in accSuper*100])

#     print('AccSuperProto = %.2f%%' % (np.mean(accSuperProto)*100), end=' ')
#     print(['%.2f%%' % d for d in accSuperProto*100])
    
    return np.mean([np.mean(accSuper), np.mean(accSuperProto)])

CIMRangePool = np.arange(1,11,dtype='int')*1000
NQuantPool = np.arange(1,65)

NPop = 128
NSel = 48
NNew = 32
maxIter = 20

iter = 0
population = np.zeros((NPop,6),dtype='int')
for N in range(NPop):
    population[N,:3] = np.random.choice(CIMRangePool,3)
    population[N,3:] = np.random.choice(NQuantPool,3)
    
bestSamples = []
while iter <= maxIter:
    iter += 1
    start_t = time.time()
    # p = Pool(processes=10)
    p = Pool()
    fitness = np.array(p.map(run_sample,[p for p in population]))
    p.close()
    
    print('Took %.2f seconds' % (time.time() - start_t))
    
#     print('All results:')
#     for N in range(NPop):
#         print(population[N], fitness[N])
    
    parents = np.argsort(-fitness)[:NSel]
    print('Top results:')
    for p in parents[:10]:
        print(population[p], fitness[p])
        
    bestSamples.append(population[parents[0]])
    
    couples = np.random.choice(parents,(NPop-NNew,2))
    oldPop = np.copy(population)
    for i,c in enumerate(couples):
        select = np.random.choice((0,1),6)
        population[i,select==0] = oldPop[c[0],select==0]
        population[i,select==1] = oldPop[c[1],select==1]
        
        mutIdx = np.random.choice([-2,-1,0,1,2],6,p=[.02,.08,.8,.08,.02])
        
        for idx in range(3):
            CIMRangeIdx = np.argwhere(CIMRangePool == population[i,idx])
            CIMRangeIdx += mutIdx[idx]
            if CIMRangeIdx < 0:
                CIMRangeIdx = 0
            elif CIMRangeIdx > len(CIMRangePool) - 1:
                CIMRangeIdx = len(CIMRangePool) - 1
            population[i,idx] = CIMRangePool[CIMRangeIdx]
            
        for idx in range(3,6):
            NQuantIdx = np.argwhere(NQuantPool == population[i,idx])
            NQuantIdx += mutIdx[idx]
            if NQuantIdx < 0:
                NQuantIdx = 0
            elif NQuantIdx > len(NQuantPool) - 1:
                NQuantIdx = len(NQuantPool) - 1
            population[i,idx] = NQuantPool[NQuantIdx]         

    for i in range(NPop-NNew,NPop):
        population[i,:3] = np.random.choice(CIMRangePool,3)
        population[i,3:] = np.random.choice(NQuantPool,3)
    
    learnedPop = []
    for i in range(NPop):
        if tuple(population[i]) in learnedPop:
            population[i,:3] = np.random.choice(CIMRangePool,3)
            population[i,3:] = np.random.choice(NQuantPool,3)
        learnedPop.append(tuple(population[i]))
    
#     for i in range(NPop):
#         mutIdx = np.random.choice([-2,-1,0,1,2],6,p=[.03,.1,.74,.1,.03])
        
#         for idx in range(3):
#             CIMRangeIdx = np.argwhere(CIMRangePool == population[i,idx])
#             CIMRangeIdx += mutIdx[idx]
#             if CIMRangeIdx < 0:
#                 CIMRangeIdx = 0
#             elif CIMRangeIdx > len(CIMRangePool) - 1:
#                 CIMRangeIdx = len(CIMRangePool) - 1
#             population[i,idx] = CIMRangePool[CIMRangeIdx]
            
#         for idx in range(3,6):
#             NQuantIdx = np.argwhere(NQuantPool == population[i,idx])
#             NQuantIdx += mutIdx[idx]
#             if NQuantIdx < 0:
#                 NQuantIdx = 0
#             elif NQuantIdx > len(NQuantPool) - 1:
#                 NQuantIdx = len(NQuantPool) - 1
#             population[i,idx] = NQuantPool[NQuantIdx]    
    
with open('imu_encoding_genetic.pickle', 'wb') as f:
    pickle.dump(bestSamples, f, protocol=pickle.HIGHEST_PROTOCOL)