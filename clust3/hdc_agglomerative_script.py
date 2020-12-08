### imports and setup

# general purpose
import numpy as np
from itertools import combinations
import pickle

# calculations
from sklearn.metrics import pairwise_distances
from sklearn import metrics
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import linkage, fcluster

# clustering
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, AffinityPropagation

### getting script argument for which gesture to run
import sys
g = int(sys.argv[1]) # gesture 0-12

### general purpose HD functions
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

### cluster relabeling functions
def remove_singleton_clusters(clusterLabels):
    X = np.copy(clusterLabels)
    clusts = np.unique(X)
    for c in clusts:
        if len(X[X == c]) == 1:
            X[X == c] = -1.0
    clusts = np.unique(X[X != -1])
    for i,c in enumerate(clusts):
        X[X == c] = i
    return X

def make_outliers_singleton(clusterLabels):
    X = np.copy(clusterLabels)
    numOutliers = len(X[X == -1])
    X[X == -1] = np.arange(numOutliers) + max(X) + 1
    return X

### clustering functions
def clust_hdc(Y,t,perm=None):
    label = -np.ones(Y.shape[0],dtype='int32')
    clusts = []
    
    if perm is not None:
        X = np.copy(Y[perm])
    else:
        X = np.copy(Y)
    
    for i,x in enumerate(X):
        if not clusts:
            clusts.append(0)
            label[i] = 0
        else:
            sim = np.zeros(len(clusts))
            for c in clusts:
                proto = bipolarize(np.sum(X[label==c],axis=0)).reshape(1,-1)
                sim[c] = 1 - cdist(proto,x.reshape(1,-1),'hamming')
            if np.max(sim) > t:
                label[i] = np.argmax(sim)
            else:
                label[i] = max(clusts) + 1
                clusts.append(max(clusts) + 1)
                
    if perm is not None:
        return label[np.argsort(perm)]
    else:
        return label

def clust_affinity(X,prefScale,damping):
    affinity = -pairwise_distances(X,metric='hamming')
    pref = prefScale*(np.min(affinity) - np.median(affinity)) + np.median(affinity)
    cl = AffinityPropagation(damping=damping,
                             max_iter=500,
                             convergence_iter=15,
                             copy=True,
                             preference=pref,
                             affinity='precomputed',
                             verbose=0).fit(affinity)
    return cl.labels_

def clust_kmeans_noinit(X,n_clusters):
    cl = KMeans(n_clusters=n_clusters,
                init='k-means++',
                n_init=100,
                max_iter=1000,
                tol = 1e-6,
                precompute_distances='auto',
                verbose=0,
                random_state=None,
                copy_x=True,
                n_jobs=-1,
                algorithm='auto').fit(X)
    return cl.labels_

def clust_kmeans_init(X,n_clusters,init_clusters):
    cl = KMeans(n_clusters=n_clusters,
                init=init_clusters,
                n_init=1,
                max_iter=1000,
                tol = 1e-6,
                precompute_distances='auto',
                verbose=0,
                random_state=None,
                copy_x=True,
                n_jobs=-1,
                algorithm='auto').fit(X)
    return cl.labels_


### generating labels
def labels_hdc(X,verbose=False, random=False):
    
    numIter = 100
    labels = {}
    
    tUpper = 1
    
    tMax = 1
    tMin = 0.5
    t = tMin
    n = 0
    
    lowestClust = X.shape[0] + 1
    if random:
        for i in range(numIter):
            if verbose:
                print("\rRuning random permutation %d out of %d" % (i+1, numIter), end =" ") 
            temp = clust_hdc(X,t,np.random.permutation(X.shape[0]))
            if len(np.unique(temp)) < lowestClust:
                lab = temp
    else:
        lab = clust_hdc(X,t)
    
    labels[t] = lab

    clusts = np.unique(lab)
    numClust = len(clusts)
    outliers = [c for c in clusts if len(lab[lab==c]) == 1]
    numNonSingle = numClust - len(outliers)
    
    if verbose:
        if random:
            print(', threshold = %f (iteration %d): %d total clusters, %d non-singleton' % (t, n, numClust, numNonSingle))
        else:
            print('Iteration %d, threshold = %f: %d total clusters, %d non-singleton' % (n, t, numClust, numNonSingle))

    while ((numNonSingle != 2) or (numClust > round(X.shape[0]/2))) and (n < 10):
        n += 1
        if numNonSingle == 1:
            tMin = t
            t = np.mean([t,tMax])
        else:
            tMax = t
            t = np.mean([t,tMin])
            
        lowestClust = X.shape[0] + 1
        if random:
            if verbose:
                print("\rRuning random permutation %d out of %d" % (i+1, numIter), end =" ") 
            for i in range(numIter):
                temp = clust_hdc(X,t,np.random.permutation(X.shape[0]))
                if len(np.unique(temp)) < lowestClust:
                    lab = temp
        else:
            lab = clust_hdc(X,t)
        
        labels[t] = lab

        clusts = np.unique(lab)
        numClust = len(clusts)
        outliers = [c for c in clusts if len(lab[lab==c]) == 1]
        numNonSingle = numClust - len(outliers)
        
        if verbose:
            if random:
                print(', threshold = %f (iteration %d): %d total clusters, %d non-singleton' % (t, n, numClust, numNonSingle))
            else:
                print('Iteration %d, threshold = %f: %d total clusters, %d non-singleton' % (n, t, numClust, numNonSingle))
            
        if numClust > X.shape[0]/10:
            tUpper = min(tUpper, t)
            
    N = 20
    for t in np.linspace(t + (tUpper-t)/(N+1),tUpper,N,endpoint=False):
        n += 1
        lowestClust = X.shape[0] + 1
        if random:
            for i in range(numIter):
                if verbose:
                    print("\rRuning random permutation %d out of %d" % (i+1, numIter), end =" ") 
                temp = clust_hdc(X,t,np.random.permutation(X.shape[0]))
                if len(np.unique(temp)) < lowestClust:
                    lab = temp
        else:
            lab = clust_hdc(X,t)
        
        labels[t] = lab

        clusts = np.unique(lab)
        numClust = len(clusts)
        outliers = [c for c in clusts if len(lab[lab==c]) == 1]
        numNonSingle = numClust - len(outliers)
        
        if verbose:
            if random:
                print(', threshold = %f (iteration %d): %d total clusters, %d non-singleton' % (t, n, numClust, numNonSingle))
            else:
                print('Iteration %d, threshold = %f: %d total clusters, %d non-singleton' % (n, t, numClust, numNonSingle))
        
        if numClust > len(X)/10:
            break
            
    return labels

def labels_affinity_prop(X,verbose=False):
    
    labels = {}
    
    for d in [0.5, 0.7, 0.9, 0.95, 0.97, 0.98, 0.99]:
        currPref = 0
        n = 0
        lab = clust_affinity(X,currPref,d)
        labels[(d,currPref)] = lab

        clusts = np.unique(lab)
        numClust = len(clusts)
        outliers = [c for c in clusts if len(lab[lab==c]) == 1]
        numNonSingle = numClust - len(outliers)
        
        if verbose:
            print('Iteration %d, preference scale = %d, damping = %f: %d total clusters, %d non-singleton' % (n, currPref, d, numClust, numNonSingle))

        currPref = 1
        n = 1
        lab = clust_affinity(X,currPref,d)
        labels[(d,currPref)] = lab

        clusts = np.unique(lab)
        numClust = len(clusts)
        outliers = [c for c in clusts if len(lab[lab==c]) == 1]
        numNonSingle = numClust - len(outliers)
        
        if verbose:
            print('Iteration %d, preference scale = %d, damping = %f: %d total clusters, %d non-singleton' % (n, currPref, d, numClust, numNonSingle))

        while numNonSingle > 1:
            n += 1
            currPref = currPref*2
            lab = clust_affinity(X,currPref,d)
            labels[(d,currPref)] = lab

            clusts = np.unique(lab)
            numClust = len(clusts)
            outliers = [c for c in clusts if len(lab[lab==c]) == 1]
            numNonSingle = numClust - len(outliers)
            
            if verbose:
                print('Iteration %d, preference scale = %d, damping = %f: %d total clusters, %d non-singleton' % (n, currPref, d, numClust, numNonSingle))

        step = currPref/4
        while step >= 0.1:
            if numNonSingle == 1:
                currPref -= step
            else:
                currPref += step
            n += 1

            lab = clust_affinity(X,currPref,d)
            labels[(d,currPref)] = lab  

            clusts = np.unique(lab)
            numClust = len(clusts)
            outliers = [c for c in clusts if len(lab[lab==c]) == 1]
            numNonSingle = numClust - len(outliers)
            
            if verbose:
                print('Iteration %d, preference scale = %f, damping = %f: %d total clusters, %d non-singleton' % (n, currPref, d, numClust, numNonSingle))

            step /= 2
            
    return labels

def labels_agglomerative(X,method,metric):
    if metric == 'hamming':
        Z = linkage(pdist(X,metric='hamming'),method=method,metric=metric)
    else:
        Z = linkage(X,method=method,metric=metric)
    lab = {}
    for n in range(len(X)):
        lab[n+1] = fcluster(Z,n+1,criterion='maxclust') - 1
    return lab

def labels_kmeans(X,init_labels=None,verbose=False):
    maxClust = round(len(X)/10)
    labels = {}
    if init_labels is not None:
        for n in range(1,maxClust+1):
            ic = centroids(X,label=init_labels[n])
            labels[n] = clust_kmeans_init(X,n,ic)
            if verbose:
                lab = labels[n]
                clusts = np.unique(lab)
                numClust = len(clusts)
                outliers = [c for c in clusts if len(lab[lab==c]) == 1]
                numNonSingle = numClust - len(outliers)
                print('Running k-means with k = %d: got %d non-single clusters' % (n, numNonSingle))
    else:
        for n in range(1,maxClust+1):
            labels[n] = clust_kmeans_noinit(X,n)
            if verbose:
                lab = labels[n]
                clusts = np.unique(lab)
                numClust = len(clusts)
                outliers = [c for c in clusts if len(lab[lab==c]) == 1]
                numNonSingle = numClust - len(outliers)
                print('Running k-means with k = %d: got %d non-single clusters' % (n, numNonSingle))
    return labels


### loading data
# select dataset and encoding type
dataName = 'allHV.npz'
emgHVType =  'hvRelAcc'

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

### run clustering
filt = gestLabel == g
X = hv[filt]

# l = labels_affinity_prop(X,verbose=True)
# with open('./clusters/labels_'+str(g)+'_affinity_prop.pickle','wb') as f:
#     pickle.dump(l,f,pickle.HIGHEST_PROTOCOL)

# l = labels_hdc(X,verbose=True,random=False)
# with open('./clusters/labels_'+str(g)+'_hdc.pickle','wb') as f:
#     pickle.dump(l,f,pickle.HIGHEST_PROTOCOL)

# l = labels_hdc(X,verbose=True,random=True)
# with open('./clusters/labels_'+str(g)+'_hdc_best.pickle','wb') as f:
#     pickle.dump(l,f,pickle.HIGHEST_PROTOCOL)

# l = labels_agglomerative(X,'ward','euclidean')
# with open('./clusters/labels_'+str(g)+'_agglomerative_w_e.pickle','wb') as f:
#     pickle.dump(l,f,pickle.HIGHEST_PROTOCOL)

# # with open('./clusters/labels_'+str(g)+'_test.pickle','wb') as f:
# #     pickle.dump(l,f,pickle.HIGHEST_PROTOCOL)

# l = labels_agglomerative(X,'centroid','euclidean')
# with open('./clusters/labels_'+str(g)+'_agglomerative_c_e.pickle','wb') as f:
#     pickle.dump(l,f,pickle.HIGHEST_PROTOCOL)

# l = labels_agglomerative(X,'median','euclidean')
# with open('./clusters/labels_'+str(g)+'_agglomerative_m_e.pickle','wb') as f:
#     pickle.dump(l,f,pickle.HIGHEST_PROTOCOL)

l = labels_agglomerative(X,'ward','hamming')
with open('./clustOut/g' + str(g) + '_clusters.pickle','wb') as f:
    pickle.dump(l,f,pickle.HIGHEST_PROTOCOL)

# l = labels_agglomerative(X,'centroid','hamming')
# with open('./clusters/labels_'+str(g)+'_agglomerative_c_h.pickle','wb') as f:
#     pickle.dump(l,f,pickle.HIGHEST_PROTOCOL)

# l = labels_agglomerative(X,'median','hamming')
# with open('./clusters/labels_'+str(g)+'_agglomerative_m_h.pickle','wb') as f:
#     pickle.dump(l,f,pickle.HIGHEST_PROTOCOL)

# l = labels_kmeans(X,verbose=True)
# with open('./clusters/labels_'+str(g)+'_kmeans.pickle','wb') as f:
#     pickle.dump(l,f,pickle.HIGHEST_PROTOCOL)

