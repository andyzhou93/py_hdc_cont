import numpy as np
import pickle

import multiprocessing
from joblib import Parallel, delayed

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist, pdist

import time

def bipolarize(Y):
    X = np.copy(Y)
    X[X > 0] = 1.0
    X[X < 0] = -1.0
    X[X == 0] = np.random.choice([-1.0, 1.0], size=len(X[X == 0]))
    return X

def centroids(X,label=None):
    if label is not None:
        cLabel,cCounts = np.unique(label,return_counts=True)
        cLabel = cLabel[cCounts > 1]
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

def test_clustering(X,y,grp,clust,numClust,numSplit=10):
    c = np.hstack([clust[l][numClust[l]] for l in np.unique(y)])
    skf = StratifiedKFold(n_splits=numSplit)
    splitIdx = 0
    acc = np.zeros(numSplit)
    for trainIdx, testIdx in skf.split(X,grp):
#         print('Running iteration %d of %d...' % (splitIdx+1, numSplit))
        XTrain, XTest = X[trainIdx], X[testIdx]
        yTrain, yTest = y[trainIdx], y[testIdx]
        cTrain, cTest = c[trainIdx], c[testIdx]
        
        AM = []
        AMlabels = []
        for l in np.unique(yTrain):
            AM.append(centroids(XTrain[yTrain == l],label=cTrain[yTrain == l])[1])
            AMlabels.append(l*np.ones(len(np.unique(cTrain[yTrain == l]))))
        AM = np.vstack(AM)
        AMlabels = np.hstack(AMlabels)
        
        pred = AMlabels[classify(XTest,AM,'hamming')]
        acc[splitIdx] = accuracy_score(pred,yTest)

        splitIdx += 1
        
    return np.mean(acc)


if __name__ == "__main__":
    np.set_printoptions(precision=4)

    ### loading data
    # select dataset and encoding type
    dataName = 'allHV.npz'
    emgHVType =  'hvRel'

    allHV = np.load(dataName)

    # extract data and labels based on gesture, trial, and position
    hv = allHV[emgHVType]
    gestLabel = allHV['gestLabel']
    posLabel = allHV['posLabel']
    trialLabel = allHV['trialLabel']

    combGP, groupGP = np.unique(np.column_stack((gestLabel,posLabel)),axis=0,return_inverse=True)
    combGPT, groupGPT = np.unique(np.column_stack((gestLabel,posLabel,trialLabel)),axis=0,return_inverse=True)

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

    maxClust = numGestures*numPositions*3
    clustering = []
    for g in gestures:
        with open('./clustOut/g' + str(g) + '_clusters.pickle','rb') as f:
            cSingles = pickle.load(f)
            cNonSingles = {}
            idx = 1
            for n in range(1,maxClust+1):
                while sum(np.unique(cSingles[idx],return_counts=True)[1] > 1) < n:
                    idx += 1
                cNonSingles[n] = cSingles[idx]
            clustering.append(cNonSingles)

    numClust = np.ones(numGestures).astype('int')
    maxClust = numGestures*numPositions*3
    res = {}

    res[sum(numClust)] = (np.copy(numClust), test_clustering(hv,gestLabel,groupGP,clustering,numClust))
    print(sum(numClust), res[sum(numClust)])
    
    while sum(numClust) <= maxClust:

        # bestAcc = 0
        # bestClust = []
        # for g in gestures:
        #     testClust = np.copy(numClust)
        #     testClust[g] += 1
        #     acc = test_clustering(hv,gestLabel,groupGP,clustering,testClust)
        #     print('\tAdding cluster to gesture %d: %f' % (g, acc))
        #     if acc > bestAcc:
        #         bestAcc = acc
        #         bestClust = testClust
        # numClust = np.copy(bestClust)
        # res[sum(numClust)] = (np.copy(numClust), bestAcc)
        st = time.time()

        testClusts = []
        for g in gestures:
            x = np.copy(numClust)
            x[g] += 1
            testClusts.append(x)

        accs = Parallel(n_jobs=-1)(delayed(test_clustering)(hv,gestLabel,groupGP,clustering,x) for x in testClusts)
        # print(accs)
        accs = np.array(accs)

        accs[numClust >= numPositions*3] = -1
        percentClusts = numClust/sum(numClust)
        accs[percentClusts > 2/numGestures] = -1

        bestAddition = np.argmax(accs)
        bestAcc = np.max(accs)

        numClust[bestAddition] += 1
        res[sum(numClust)] = (np.copy(numClust), bestAcc)

        et = time.time()
        print(sum(numClust), res[sum(numClust)], et-st)

        if sum(numClust) % numGestures == 0:
            with open('optimal_clustering_results_' + str(sum(numClust)) + '.pickle', 'wb') as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('optimal_clustering_results.pickle', 'wb') as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
