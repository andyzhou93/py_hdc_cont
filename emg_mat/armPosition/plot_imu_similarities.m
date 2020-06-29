close all
clear
clc

figure
set(gcf,'position',[300 300 1500 600])

%% feature similarities
load('sub1exp1.mat');
[numGest, numPos, numTrial] = size(experimentData);
posVals = zeros(3,numPos);
for p = 1:numPos
    for g = 1:numGest
        for t = 1:numTrial
            posVals(:,p) = posVals(:,p) + sum(experimentData(g,p,t).accMeanFeat(experimentData(g,p,t).expGestLabel > 0,:))';
        end
    end
end
posVals = posVals./80/numGest/numTrial;
get_dist(posVals)
subplot(1,2,1)
imagesc(get_dist(posVals),[-1 1])
colorbar
axis square

%% encoder similarities
load('sub1exp1_hv.mat')
posIM = zeros(10000,numPos);
for p = 1:numPos
    for g = 1:numGest
        for t = 1:numTrial
            posIM(:,p) = posIM(:,p) + sum(double(experimentData(g,p,t).accHV64(:,experimentData(g,p,t).expGestLabel > 0)),2);
        end
    end
end
posIM(posIM >= 0) = 1;
posIM(posIM < 0) = -1;
get_dist(posIM)
subplot(1,2,2)
imagesc(get_dist(posIM),[0 1])
colorbar
axis square

%%
function [sims] = get_dist(cim)
    sims = (cim'*cim)./(vecnorm(cim)'*vecnorm(cim));
end