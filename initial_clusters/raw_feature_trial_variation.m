close all
clearvars -except experimentData
clc

if ~exist('experimentData','var')
    tic
    load('../emg_mat/armPosition/sub1exp1.mat')
    toc
end

[numGest,numPos,numTrial] = size(experimentData);

for g = 1:numGest
    for p = 1:numPos
        figure(1)
        for t = 1:numTrial
            subplot(2,numTrial,t)
            plot(experimentData(g,p,t).emgRaw);
            subplot(2,numTrial,t+numTrial)
            plot(experimentData(g,p,t).emgFeat);
        end
    end
end