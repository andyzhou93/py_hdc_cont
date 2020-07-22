close all
clearvars -except experimentData
clc

if ~exist('experimentData','var')
    tic
    load('../emg_mat/armPosition/sub1exp0_hv.mat')
    toc
end

[numGest,numPos,numTrial] = size(experimentData);
D = size(experimentData(1,1,1).emgHV,1);

for g = 1:numGest
    for p = 1:numPos
        for t = 1:numTrial
            hv = double(experimentData(g,p,t).emgHV(:,experimentData(g,p,t).expGestLabel > 0));
            elementMajority = sum(hv,2);
            elementCorrelation = hv*hv';
            numHV = size(hv,2);
            for n = 0:2:numHV
                idx = find(abs(elementMajority) == n);
                figure(1)
                histogram(elementCorrelation(idx,idx) + diag(nan(length(idx),1)));
            end
        end
    end
end