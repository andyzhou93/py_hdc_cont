from ipyparallel import Client
import scipy.io as sio
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.metrics import accuracy_score

import hdc
import time

import sys
numIter = int(sys.argv[1])

def runIterSeeded(seed):
    import numpy as np
    import pandas as pd
    from sklearn import svm
    from sklearn.metrics import accuracy_score

    import sys
    # sys.path.append('/Users/andy/Research/py_hdc_cont/hdc_scripts')
    sys.path.append('/global/home/users/andyz/py_hdc_cont/hdc_scripts')

    import hdc

    np.random.seed(seed)

    isTrain = np.empty(0)
    testTrials = np.random.randint(0,numTrial,numGest)
    for g in range(numGest):
        for t in range(numTrial):
            if t != testTrials[g]:
                isTrain = np.concatenate((isTrain,np.random.permutation(np.concatenate((np.ones(int(round(tp*numEx))), -np.ones(numEx - int(round(tp*numEx))))))))
            else:
                isTrain = np.concatenate((isTrain,np.zeros(numEx)))
    testTrials = np.random.randint(0,numTrial,numGest)
    for g in range(numGest):
        for t in range(numTrial):
            if t != testTrials[g]:
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

client = Client()
dview = client[:]

# segment only the hold portions of data
holdStart = 70
holdEnd = 149
numEx = holdEnd - holdStart + 1

# hypervector and feature dimensions
D = 10000
numFeat = 320
numGest = 13
numTrial = 5

# testPercentage = np.linspace(0.05,1,20)
# adaptThreshold = np.linspace(0.05,0.8,16)
testPercentage = np.linspace(0.1,1,2)
adaptThreshold = np.linspace(0.2,0.65,2)

numSVM = np.zeros((5,len(testPercentage),numIter))
accSVM = np.zeros((5,len(testPercentage),numIter))

numHDC = np.zeros((5,len(testPercentage),len(adaptThreshold),numIter))
accHDC = np.zeros((5,len(testPercentage),len(adaptThreshold),numIter))

# subject labels are 1-indexed
# for s in range(5):
for s in [1]:
    subject = s + 1

    runData = {}
    runData['numTrial'] = numTrial
    runData['numGest'] = numGest
    runData['numEx'] = numEx
    runData['adaptThreshold'] = adaptThreshold
    runData['D'] = D
    runData['numFeat'] = numFeat
    runData['subject'] = subject
    dview.push(runData,block=True)

    # loop through different testing percentages
    for tpIdx,tp in enumerate(testPercentage):
        dview.push({'tp':tp},block=True)

        lview = client.load_balanced_view()
        lview.block = True

        print('Running with %f of single trial for training' % (tp))
        startTime = time.time()
        res = lview.map(runIterSeeded, range(numIter))
        stopTime = time.time()
        print('\tTook %f seconds total, %f per iteration' % ((stopTime - startTime), (stopTime - startTime)/numIter))

        for i in range(numIter):
            accSVM[s,tpIdx,i] = res[i][2]
            numSVM[s,tpIdx,i] = res[i][3]
            accHDC[s,tpIdx,:,i] = res[i][0]
            numHDC[s,tpIdx,:,i] = res[i][1]

matOut = {}
matOut['accSVM'] = accSVM
matOut['accHDC'] = accHDC
matOut['numSVM'] = numSVM
matOut['numHDC'] = numHDC
sio.savemat('four_trial_adapt_' + str(numIter) + 'iters.mat',matOut)
