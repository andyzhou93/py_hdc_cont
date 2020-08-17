# general purpose
import numpy as np
from itertools import combinations
import pickle

# calculations
from sklearn.metrics import pairwise_distances
from sklearn import metrics
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import linkage, fcluster

import sys
g = int(sys.argv[1]) # gesture 0-12

def bipolarize(Y):
    X = np.copy(Y)
    X[X > 0] = 1.0
    X[X < 0] = -1.0
    X[X == 0] = np.random.choice([-1.0, 1.0], size=len(X[X == 0]))
    return X

def centroids(X,label=None,doMean=False):
    if label is not None:
        c = np.zeros((len(np.unique(label)), X.shape[1]))
        for i,l in enumerate(np.unique(label)):
            if doMean:
                c[i,:] = np.mean(X[label==l],axis=0)
            else:
                c[i,:] = bipolarize(np.sum(X[label==l],axis=0))
    else:
        if doMean:
            c = np.mean(X,axis=0).reshape(1,-1)
        else:
            c = bipolarize(np.sum(X,axis=0)).reshape(1,-1)
    return c

def num_nonsingle(X):
    _,counts = np.unique(X,return_counts=True)
    return len(counts[counts != 1])

def clust_hdc(Y,t,maxClust,perm=None,earlyStop=False):
    label = -np.ones(len(Y),dtype='int32')
    clusts = []
    
    if perm is not None:
        X = np.copy(Y[perm])
    else:
        X = np.copy(Y)
    
    for i,x in enumerate(X):
        _,counts = np.unique(label[label != -1], return_counts=True)
        numClust = len(counts[counts != 1])
        if (numClust >= maxClust) and earlyStop:
            break
        elif not clusts:
            clusts.append(0)
            label[i] = 0
        else:
            sim = np.zeros(len(clusts))
            for c in clusts:
                proto = bipolarize(np.sum(X[label==c],axis=0)).reshape(1,-1)
                sim[c] = 1 - cdist(proto,x.reshape(1,-1),'hamming')
            if (np.max(sim) > t) or (numClust >= maxClust) :
                label[i] = np.argmax(sim)
            else:
                label[i] = max(clusts) + 1
                clusts.append(max(clusts) + 1)
        
                
    if perm is not None:
        return label[np.argsort(perm)]
    else:
        return label

def find_best_thresh(X,targetClusts):
    print('Searching for threshold that gives %d clusters' % targetClusts)
    tMax = 1
    tMin = 0.5
    tNew = 0.5
    tFound = False
    
    while tMax - tMin > 0.00005:
        tCurr = tNew
        l = clust_hdc(X,tCurr,maxClust=targetClusts,perm=None,earlyStop=True)
        if -1 in l: # too many clusters, stopped early
            print('t = %f - early stop' % tCurr)
            tMax = tCurr
            tNew = np.mean([tCurr,tMin])
        else:
            _,counts = np.unique(l, return_counts=True)
            numClust = len(counts[counts != 1])
            print('t = %f - n = %d' % (tCurr,numClust))
            if numClust > targetClusts:
                tMax = tCurr
                tNew = np.mean([tCurr,tMin])
            elif numClust <= targetClusts:
                tMin = tCurr
                tNew = np.mean([tCurr,tMax])
                if numClust == targetClusts:
                    tBest = tCurr
                    lBest = l
                    tFound = True
                    
    if tFound:
        print('Best guess: %f with %d clusters' % (tBest, targetClusts))
        print('')
        return tBest, lBest
    else:
        print('Best guess: %f with %d clusters' % (tCurr, numClust))
        print('')
        return tCurr, l




# select dataset and encoding type
dataName = 'allHV.npz'
emgHVType =  'hvRel'

allHV = np.load(dataName)

# extract data and labels based on gesture, trial, and position
hv = allHV[emgHVType]
gestLabel = allHV['gestLabel']
posLabel = allHV['posLabel']
trialLabel = allHV['trialLabel']

# get list of unique values for each label
gestures = np.unique(gestLabel)
positions = np.unique(posLabel)
trials = np.unique(trialLabel)

numGestures = len(gestures)
numPositions = len(positions)
numTrials = len(trials)

# get data size info
D = hv.shape[1] # hypervector dimension
numHV = 80 # number of examples per trial

standardOrder = []
filt = (gestLabel == g)
X = np.copy(hv[filt])
posFilt = posLabel[filt]

tMin = find_best_thresh(X,2)[0]
tMax = find_best_thresh(X,32)[0]
tRange = np.linspace(tMin,tMax,30)

nIter = 30
standardOrder = []
randPos = []
randAll = []

for n in range(nIter):
    allPerm = np.random.permutation(len(X))
    
    posPerm = np.arange(len(X))
    for p in positions:
        np.random.shuffle(posPerm[np.where(posFilt==p)[0][0]:(np.where(posFilt==p)[0][-1] + 1)])
    posPerm = posPerm.reshape(-1,round(len(X)/numPositions))
    np.random.shuffle(posPerm)
    posPerm = posPerm.flatten()
    
    a = []
    b = []
    for t in tRange:
        print('Running iteration %d: thresh = %f' % (n,t))
        if n == 0:
            standardOrder.append(clust_hdc(X,t,maxClust=len(X),perm=None))
            
        a.append(clust_hdc(X,t,maxClust=len(X),perm=allPerm))
        b.append(clust_hdc(X,t,maxClust=len(X),perm=posPerm))
        
    randPos.append(b)
    randAll.append(a)

res = {'tRange':tRange,'standardOrder':standardOrder,'randPos':randPos,'randAll':randAll}
with open('./clusters/hdc_run_'+str(g)+'.pickle','wb') as f:
    pickle.dump(res,f,pickle.HIGHEST_PROTOCOL)
