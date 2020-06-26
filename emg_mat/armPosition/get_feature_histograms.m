close all
clear
clc

load('sub1exp0.mat');

%%
[numGest, numPos, numTrial] = size(experimentData);
featValues = zeros(numGest,numPos,64);

for g = 1:numGest
    for p = 1:numPos
        feat = [];
        for t = 1:numTrial
            feat = [feat; experimentData(g,p,t).emgFeat(111:190,:)];
        end
        featValues(g,p,:) = mean(feat);
    end
end

maxFeat = squeeze(max(featValues,[],1));
minFeat = squeeze(min(featValues,[],1));
rangeFeat = maxFeat - minFeat;

bar(maxFeat);
hold on
bar(minFeat,'FaceColor',[1 1 1]);