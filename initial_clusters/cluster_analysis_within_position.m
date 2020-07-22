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

numIter = 10;

for g = 1
    for p = 3
        trial = ([]);
        allHV = [];
        pts = zeros(D,numTrial);
        ptMags = zeros(D,numTrial);
        for t = 1:numTrial
            hv = double(experimentData(g,p,t).emgHV(:,experimentData(g,p,t).expGestLabel > 0));
            numHV = size(hv,2);
            pt = bipolarize_hv(sum(hv,2));
            pairSims = get_cosine_sim(hv,hv);
            protoSims = get_cosine_sim(hv,pt);
            meanPairSims = mean(pairSims(triu(ones(size(pairSims)),1)==1));
            minPairSims = min(pairSims(triu(ones(size(pairSims)),1)==1));
            maxPairSims = max(pairSims(triu(ones(size(pairSims)),1)==1));
            medPairSims = median(pairSims(triu(ones(size(pairSims)),1)==1));
            meanProtoSims = mean(protoSims);
            minProtoSims = min(protoSims);
            maxProtoSims = max(protoSims);
            medProtoSims = median(protoSims);
            ptMag = sum(hv,2)./max(abs(sum(hv,2)));
            uIdx = find(abs(ptMag) > 0.7);
            
            trial(t).hv = hv;
            trial(t).numHV = numHV;
            trial(t).pt = pt;
            trial(t).pairSims = pairSims;
            trial(t).protoSims = protoSims;
            trial(t).pairSimsStats = [minPairSims meanPairSims medPairSims maxPairSims];
            trial(t).protoSimsStats = [minProtoSims meanProtoSims medProtoSims maxProtoSims];
            trial(t).ptMag = ptMag;
            
            allHV = [allHV hv];
            pts(:,t) = pt;
            ptMags(:,t) = ptMag;
        end
        
        figure(1)
        
        numSp = 7;
        numSpCols = 6;
        numSpRows = ceil(numSp/numSpCols);
        set(gcf,'position',[1 1 2048 256*numSpRows])
        spIdx = 1;
        
        subplot(numSpRows,numSpCols,spIdx)
        for t = 1:numTrial
            histogram(trial(t).ptMag,'Normalization','probability')
            hold on
        end
        hold off
        ylim([0 1])
        spIdx = spIdx + 1;
        
        subplot(numSpRows,numSpCols,spIdx)
        histogram(sum(ptMags,2)./3,'Normalization','probability')
        ylim([0 1])
        spIdx = spIdx + 1;
        
        subplot(numSpRows,numSpCols,spIdx)
        for t = 1:numTrial
            histogram(trial(t).ptMag,-1:0.25:1,'Normalization','probability')
            hold on
        end
        hold off
        ylim([0 1])
        spIdx = spIdx + 1;
        
        subplot(numSpRows,numSpCols,spIdx)
        histogram(sum(ptMags,2)./3,-1:0.25:1,'Normalization','probability')
        ylim([0 1])
        spIdx = spIdx + 1;
        
        subplot(numSpRows,numSpCols,spIdx)
        heatmap(get_cosine_sim(pts,pts),'ColorLimits',[0 1]);
        spIdx = spIdx + 1;
        
        subplot(numSpRows,numSpCols,spIdx)
        heatmap(get_cosine_sim(ptMags,ptMags),'ColorLimits',[0 1]);
        spIdx = spIdx + 1;
        
        for t = 1:numTrial
            sim = get_cosine_sim(pts,trial(t).hv);
            subplot(numSpRows,numSpCols,spIdx);
            plot(sim')
            ylim([0 1])
            spIdx = spIdx + 1;
        end
        
            

    end
end

%% useful functions
function [hvOut] = bipolarize_hv(hvIn)
    hvOut = hvIn;
    hvOut(hvOut > 0) = 1;
    hvOut(hvOut < 0) = -1;
    hvOut(hvOut == 0) = (randi(2,length(hvOut(hvOut==0)),1)-1.5)*2;
end

function [sims] = get_cosine_sim(a,b)
    sims = (a'*b)./(vecnorm(a)'*vecnorm(b));
end