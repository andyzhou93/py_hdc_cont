close all
clear
clc

load('sub1exp2.mat');
[numGest,numPos,numTrial] = size(experimentData);
numHV = length(find(experimentData(1,1,1).expGestLabel>0));

allAcc = zeros(numHV*numGest*numPos*numTrial,3);
posLabel = zeros(numHV*numGest*numPos*numTrial,1);
idx = 1:numHV;
for p = 1:numPos
    for g = 1:numGest
        for t = 1:numTrial
            allAcc(idx,:) = experimentData(g,p,t).accMeanFeat(experimentData(g,p,t).expGestLabel > 0,:);
            posLabel(idx) = p;
            idx = idx + numHV;
        end
    end
end

%%
accStds = zeros(numPos,3);
for p = 1:numPos
    accStds(p,:) = std(allAcc(posLabel==p,:));
end