import scipy.io as sio
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.metrics import accuracy_score

import hdc
import time

import sys
s = int(sys.argv[1])
numIter = int(sys.argv[2])

# segment only the hold portions of data
holdStart = 70
holdEnd = 149
numEx = holdEnd - holdStart + 1

# hypervector and feature dimensions
D = 10000
numFeat = 320
numGest = 13
numTrial = 5

#testPercentage = np.linspace(0.05,1,20)
testPercentage = np.linspace(0.55,0.8,6)
adaptThreshold = np.linspace(0.1,0.8,8)
#testPercentage = np.linspace(0.1,1,2)
#adaptThreshold = np.linspace(0.2,0.65,2)

numSVM = np.zeros((len(testPercentage),numIter))
accSVM = np.zeros((len(testPercentage),numIter))

numHDC = np.zeros((len(testPercentage),len(adaptThreshold),numIter))
accHDC = np.zeros((len(testPercentage),len(adaptThreshold),numIter))

subject = s + 1

def runIterSeeded(tp,seed):
    np.random.seed(seed)

    isTrain = np.empty(0)
    trainTrials = np.random.randint(0,numTrial,numGest)
    for g in range(numGest):
        for t in range(numTrial):
            if t == trainTrials[g]:
                isTrain = np.concatenate((isTrain,np.random.permutation(np.concatenate((np.ones(int(round(tp*numEx))), -np.ones(numEx - int(round(tp*numEx))))))))
            else:
                isTrain = np.concatenate((isTrain,np.zeros(numEx)))
    trainTrials = np.random.randint(0,numTrial,numGest)
    for g in range(numGest):
        for t in range(numTrial):
            if t == trainTrials[g]:
                isTrain = np.concatenate((isTrain,np.random.permutation(np.concatenate((np.ones(int(round(tp*numEx))), -np.ones(numEx - int(round(tp*numEx))))))))
            else:
                isTrain = np.concatenate((isTrain,np.zeros(numEx)))

    # featData = pd.read_feather('/Users/andy/Research/py_hdc_cont/hdc_scripts/S' + str(subject) + '_feature.df')
    # ngramData = pd.read_feather('/Users/andy/Research/py_hdc_cont/hdc_scripts/S' + str(subject) + '_ngram.df')

    featData = pd.read_feather('/global/home/users/andyz/py_hdc_cont/hdc_scripts/S' + str(subject) + '_feature.df')
    ngramData = pd.read_feather('/global/home/users/andyz/py_hdc_cont/hdc_scripts/S' + str(subject) + '_ngram.df')

    featData['isTrain'] = isTrain
    ngramData['isTrain'] = isTrain

    # train HD model
    allGest = ngramData['gesture'].unique()
    
    # loop through all HDC adaptive thresholds
    accHDC = np.zeros(len(adaptThreshold))
    numHDC = np.zeros(len(adaptThreshold))
    for atIdx,at in enumerate(adaptThreshold):
        AM = hdc.Memory(D)
        for g in allGest:
            ng = np.asarray(ngramData.loc[(ngramData['gesture'] == g) & (ngramData['isTrain'] == 1) & (ngramData['context'] == 0)].iloc[:,0:D])
            AM.train_sub_cluster(ng,vClass=g,threshold=at)
            AM.prune(min=5)

        for g in allGest:
            ng = np.asarray(ngramData.loc[(ngramData['gesture'] == g) & (ngramData['isTrain'] == 1) & (ngramData['context'] == 1)].iloc[:,0:D])
            AM.train_sub_cluster(ng,vClass=g,threshold=at)
            AM.prune(min=5)
            
        # collect testing data and perform inference
        testNgram = ngramData.loc[(ngramData['isTrain'] == 0)].iloc[:,0:D]
        testLabel = ngramData.loc[(ngramData['isTrain'] == 0)].iloc[:,D]
        label,sim = AM.match(np.asarray(testNgram),bipolar=True)
        accHDC[atIdx] = (label == np.asarray(testLabel)).sum()/len(label)
        numHDC[atIdx] = len(AM.classes)

    # train and test SVM model
    clf = svm.SVC(decision_function_shape='ovo',kernel='linear',C=1)
    clf.fit(featData.loc[featData['isTrain'] == 1].iloc[:,0:numFeat],featData.loc[featData['isTrain'] == 1].iloc[:,numFeat])
    yhat = clf.predict(featData.loc[featData['isTrain'] == 0].iloc[:,0:numFeat])
    accSVM = accuracy_score(yhat,featData.loc[featData['isTrain'] == 0].iloc[:,numFeat])
    numSVM = len(clf.support_)

    return accHDC, numHDC, accSVM, numSVM


# loop through different testing percentages
for tpIdx,tp in enumerate(testPercentage):
    print('Running with %f of single trial for training' % (tp))
    startTime = time.time()
    res = [runIterSeeded(tp,n) for n in range(numIter)]
    stopTime = time.time()
    print('\tTook %f seconds total, %f per iteration' % ((stopTime - startTime), (stopTime - startTime)/numIter))

    for i in range(numIter):
        accSVM[tpIdx,i] = res[i][2]
        numSVM[tpIdx,i] = res[i][3]
        accHDC[tpIdx,:,i] = res[i][0]
        numHDC[tpIdx,:,i] = res[i][1]

matOut = {}
matOut['accSVM'] = accSVM
matOut['accHDC'] = accHDC
matOut['numSVM'] = numSVM
matOut['numHDC'] = numHDC
sio.savemat('./output/single_trial2_adapt_sub' + str(subject) + '_' + str(numIter) + 'iters.mat',matOut)
