import scipy.io as sio
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.metrics import accuracy_score

import hdc
import time

# location of all offline data
dataDir = '../emg_mat/offline/'

# choose experiments for base and new context
baseExperiment = 1
newExperiment = 3

# segment only the hold portions of data
holdStart = 70
holdEnd = 149
numEx = holdEnd - holdStart + 1

# hypervector and feature dimensions
D = 10000
numFeat = 320


numIter = 1
# testPercentage = np.linspace(0.05,1,20)
# adaptThreshold = np.linspace(0.05,0.8,16)
testPercentage = np.linspace(0.1,0.8,2)
adaptThreshold = np.linspace(0.2,0.5,2)

numSVM = np.zeros((5,len(testPercentage),numIter))
accSVM = np.zeros((5,len(testPercentage),numIter))

numHDC = np.zeros((5,len(testPercentage),len(adaptThreshold),numIter))
accHDC = np.zeros((5,len(testPercentage),len(adaptThreshold),numIter))

# subject labels are 1-indexed
for s in range(5):
    subject = s + 1
    print('Gathering data for Subject ' + str(subject))

    # load data from the two contexts
    filename = dataDir + 'S' + str(subject) + 'E' + str(baseExperiment) + '.mat'
    base = sio.loadmat(filename)['emgHD']
    filename = dataDir + 'S' + str(subject) + 'E' + str(newExperiment) + '.mat'
    new = sio.loadmat(filename)['emgHD']

    # get metatdata
    numGest, numTrial = base.shape
    numCh = base[0,0][2].shape[1]

    # collect all data and as single dataframe
    features = np.empty((numCh*5,0))
    ngrams = np.empty((D,0))
    labels = np.empty(0)
    trials = np.empty(0)
    context = np.empty(0)

    # collect baseline data
    for g in range(numGest):
        for t in range(numTrial):
            trial = base[g,t]
            feat = np.empty((0,numEx))
            for i in range(5):
                feat = np.concatenate((feat,trial[2][(holdStart+i):(holdEnd+i+1),:].T),axis=0)
            features = np.concatenate((features,feat),axis=1)
            ngrams = np.concatenate((ngrams,trial[3][:,holdStart:holdEnd+1]),axis=1)
            labels = np.concatenate((labels,g*np.ones(numEx)))
            trials = np.concatenate((trials,t*np.ones(numEx)))
            context = np.concatenate((context,0*np.ones(numEx)))

    # collect new data
    for g in range(numGest):
        for t in range(numTrial):
            trial = new[g,t]
            feat = np.empty((0,numEx))
            for i in range(5):
                feat = np.concatenate((feat,trial[2][(holdStart+i):(holdEnd+i+1),:].T),axis=0)
            features = np.concatenate((features,feat),axis=1)
            ngrams = np.concatenate((ngrams,trial[3][:,holdStart:holdEnd+1]),axis=1)
            labels = np.concatenate((labels,g*np.ones(numEx)))
            trials = np.concatenate((trials,t*np.ones(numEx)))
            context = np.concatenate((context,1*np.ones(numEx)))

    # create dataframe for features
    featCols = ['feature' + str(i) for i in range(features.shape[0])]
    featData = pd.DataFrame(features.T,columns=featCols)
    featData['gesture'] = labels
    featData['trial'] = trials
    featData['context'] = context

    # create dataframe for ngrams
    ngramCols = ['hv' + str(i) for i in range(ngrams.shape[0])]
    ngramData = pd.DataFrame(ngrams.T,columns=ngramCols)
    ngramData['gesture'] = labels
    ngramData['trial'] = trials
    ngramData['context'] = context

    # loop through different testing percentages
    for tpIdx,tp in enumerate(testPercentage):
        print('Running with %f of single trial for training' % (tp))
        # iterate through to get averages (different cross-validation folds)
        elapsedTime = 0
        for n in range(numIter):
            startTime = time.time()
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

            featData['isTrain'] = isTrain
            ngramData['isTrain'] = isTrain

            # train HD model
            allGest = ngramData['gesture'].unique()
            
            # loop through all HDC adaptive thresholds
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
    #             AM.prune(min=5)
                label,sim = AM.match(np.asarray(testNgram),bipolar=True)
                accHDC[s,tpIdx,atIdx,n] = (label == np.asarray(testLabel)).sum()/len(label)
                numHDC[s,tpIdx,atIdx,n] = len(AM.classes)

            # train and test SVM model
            clf = svm.SVC(decision_function_shape='ovo',kernel='linear',C=1)
            clf.fit(featData.loc[featData['isTrain'] == 1].iloc[:,0:numFeat],featData.loc[featData['isTrain'] == 1].iloc[:,numFeat])
            yhat = clf.predict(featData.loc[featData['isTrain'] == 0].iloc[:,0:numFeat])
            accSVM[s,tpIdx,n] = accuracy_score(yhat,featData.loc[featData['isTrain'] == 0].iloc[:,numFeat])
            numSVM[s,tpIdx,n] = len(clf.support_)
            
            endTime = time.time()
            elapsedTime = (elapsedTime*n/(n+1)) + ((endTime-startTime)/(n+1))
            print('Finished iteration %d, average time %f seconds\r' % (n+1, elapsedTime), end="")
        print('')


matOut = {}
matOut['accSVM'] = accSVM
matOut['accHDC'] = accHDC
matOut['numSVM'] = numSVM
matOut['numHDC'] = numHDC
sio.savemat('single_trial_adapt.mat',matOut)