close all
clearvars -except experimentData
clc

if ~exist('experimentData','var')
    load('data.mat')
end

[numGest, numPos, numTrial] = size(experimentData);

am = zeros(10000,numGest*numPos*numTrial);
gLab = zeros(numGest*numPos*numTrial,1);
pLab = zeros(numGest*numPos*numTrial,1);
tLab = zeros(numGest*numPos*numTrial,1);
idx = 1;
for g = 1:numGest
    for p = 1
        for t = 1:(numTrial-1)
            hv = double(experimentData(g,p,t).emgHV(:,experimentData(g,p,t).expGestLabel > 0));
            am(:,idx) = am(:,idx) + sum(hv,2);
        end
        gLab(idx) = g;
        pLab(idx) = p;
        idx = idx + 1;
    end
end

am = bipolarize_hv(am);

numCorr = 0;
numAll = 0;
for g = 1:numGest
    for p = 1:numPos
        for t = numTrial
            hv = double(experimentData(g,p,t).emgHV(:,experimentData(g,p,t).expGestLabel > 0));
            sim = get_cosine_sim(hv,am);
            [~,i] = max(sim,[],2);
            l = gLab(i);
            numCorr = numCorr + sum(l == g);
            numAll = numAll + length(l);
        end
    end
end



%%
holdFeat = [experimentData(:,1,:).emgFeat];


figure
set(gcf,'position',[1 1 2048 1184])
ax1 = subplot(2,1,1);
histogram(holdFeat,'EdgeColor','none')
% hold on
% histogram(holdFeat(111:190,:),'EdgeColor','none')
% histogram(holdFeat([1:30 271:300],:),'EdgeColor','none')

grid on
grid minor

ax2 = subplot(2,1,2);
histogram(holdFeat,'Normalization','cdf','EdgeColor','none')
% hold on
% histogram(holdFeat(111:190,:),'Normalization','cdf','EdgeColor','none')
% histogram(holdFeat([1:30 271:300],:),'Normalization','cdf','EdgeColor','none')

linkaxes([ax1 ax2],'x')
grid on
grid minor


function [sims] = get_cosine_sim(a,b)
    sims = (a'*b)./(vecnorm(a)'*vecnorm(b));
%     D = size(a,1);
%     sims = ((a'*b) + D)./2./D;
end

function [imBi] = bipolarize_hv(im)
    numHV = size(im,2);
    imBi = zeros(size(im));
    for i = 1:numHV
        hv = im(:,i);
        hv(hv>0) = 1;
        hv(hv<0) = -1;
        numzeros = sum(hv==0);
        hv(hv==0) = gen_random_hv(numzeros);
        imBi(:,i) = hv;
    end
end

function [hv] = gen_random_hv(D)
    hv = ones(D,1);
    flip = randperm(D);
    hv(flip(1:round(D/2))) = -1;
end
