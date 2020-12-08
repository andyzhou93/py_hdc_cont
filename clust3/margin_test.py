import numpy as np
from scipy.spatial.distance import cdist, pdist
# import multiprocessing
# from joblib import Parallel, delayed
from tqdm import tqdm
import sys



def bipolarize(Y):
    X = np.copy(Y)
    X[X > 0] = 1.0
    X[X < 0] = -1.0
    X[X == 0] = np.random.choice([-1.0, 1.0], size=len(X[X == 0]))
    return X

def get_inter_dist(bits,numEx,D):
    out = np.zeros(len(bits))
    for i,NFlip in enumerate(bits):
        hv = np.zeros((numEx,D))
        seed = np.random.choice([-1.0, 1.0], size=D)
        for n in range(numEx):
            flipIdx = np.random.permutation(D)[:NFlip]
            flipBits = np.ones(D)
            flipBits[flipIdx] = -1
            hv[n,:] = seed*flipBits
        
        d = cdist(hv,hv,'hamming')
        out[i] = np.mean(d[~np.eye(d.shape[0],dtype=bool)])
    return out


def get_margin(pBits,cBits,eBits,numGestures,numPositions,D,numEx):
    marginOut = np.zeros((len(pBits),len(cBits),len(eBits)))
    for i,NFlipA in enumerate(pBits):

        allCentroids = np.zeros((numGestures,numPositions,D))
        seed = np.random.choice([-1.0, 1.0], size=D)
        for g in range(numGestures):
            for p in range(numPositions):
                flipIdx = np.random.permutation(D)[:NFlipA]
                flipBits = np.ones(D)
                flipBits[flipIdx] = -1
                allCentroids[g,p,:] = seed*flipBits
        
        for j,NFlipB in enumerate(cBits):            

            contexts = np.zeros((numPositions,D))
            seed = np.random.choice([-1.0, 1.0], size=D)
            for p in range(numPositions):
                flipIdx = np.random.permutation(D)[:NFlipB]
                flipBits = np.ones(D)
                flipBits[flipIdx] = -1
                contexts[p,:] = seed*flipBits
            
            mappedCentroids = np.zeros(allCentroids.shape)
            for g in range(numGestures):
                for p in range(numPositions):
                    mappedCentroids[g,p,:] = allCentroids[g,p,:] * contexts[p,:]

            prototypes = np.zeros((numGestures,D))
            for g in range(numGestures):
                prototypes[g,:] = bipolarize(np.sum(mappedCentroids[g,:,:],axis=0))
            
            for k,NFlipC in enumerate(eBits):

                margins = np.zeros((numGestures,numPositions,numEx))
                for g in range(numGestures):
                    for p in range(numPositions):
                        examples = np.zeros((numEx,D))
                        seed = mappedCentroids[g,p,:]
                        for ex in range(numEx):
                            flipIdx = np.random.permutation(D)[:NFlipC]
                            flipBits = np.ones(D)
                            flipBits[flipIdx] = -1
                            examples[ex,:] = seed*flipBits

                        dists = cdist(examples,prototypes,'hamming')
                        b = np.min(np.delete(dists,g,axis=1),axis=1)
                        a = dists[:,g]
                        margins[g,p,:] = (b - a)/(b + a)

                marginOut[i,j,k] = np.mean(margins)

    return marginOut

if __name__ == "__main__":

    g = int(sys.argv[1])

    D = 10000
    numGestures = 13
    numPositions = 8
    numEx = 30

    # prototypeBits = np.round(np.logspace(1,np.log10(5000),20)).astype('int')
    # contextBits = np.round(np.logspace(-0.5,np.log10(5000),20)).astype('int')
    # exampleBits = np.round(np.logspace(-0.5,np.log10(5000),20)).astype('int')

    prototypeBits = np.round(np.linspace(100,5000,20)).astype('int')
    contextBits = np.round(np.linspace(0,5000,20)).astype('int')
    exampleBits = np.round(np.linspace(0,5000,20)).astype('int')

    # prototypeBits = np.round(np.linspace(100,5000,3)).astype('int')
    # contextBits = np.round(np.linspace(0,5000,3)).astype('int')
    # exampleBits = np.round(np.linspace(0,5000,3)).astype('int')

    numIter = 2

    prototypeDistances = get_inter_dist(prototypeBits,numEx,D)
    contextDistances = get_inter_dist(contextBits,numEx,D)
    exampleDistances = get_inter_dist(exampleBits,numEx,D)

    # for ii in tqdm(range(numIter)):
    for ii in range(numIter):
        marginOut = get_margin(prototypeBits,contextBits,exampleBits,numGestures,numPositions,D,numEx)

        with open('test_%d.npy' % (ii + g*numIter), 'wb') as f:
            np.savez(f,marginOut=marginOut,prototypeDistances=prototypeDistances,contextDistances=contextDistances,exampleDistances=exampleDistances)


