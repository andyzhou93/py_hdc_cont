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

%% gather hypervector stats
hvStats = ([]);
for g = 1:numGest
    for p = 1:numPos
        for t = 1:numTrial
            hv = double(experimentData(g,p,t).emgHV(:,experimentData(g,p,t).expGestLabel > 0));
            pt = bipolarize_hv(sum(hv,2));
            pairSims = get_cosine_sim(hv,hv);
            protoSims = get_cosine_sim(hv,pt);
            meanPairSims = mean(pairSims(triu(ones(size(pairSims)),1)==1));
            minPairSims = min(pairSims(triu(ones(size(pairSims)),1)==1));
            meanProtoSims = mean(protoSims);
            minProtoSims = min(protoSims);
            ptMag = sum(hv,2)./max(abs(sum(hv,2)));
            uIdx = find(abs(ptMag) > 0.7);
            
            hvStats(g,p,t).hv = hv;
            hvStats(g,p,t).pt = pt;
            hvStats(g,p,t).pairSims = pairSims;
            hvStats(g,p,t).protoSims = protoSims;
            hvStats(g,p,t).meanPairSims = meanPairSims;
            hvStats(g,p,t).minPairSims = minPairSims;
            hvStats(g,p,t).meanProtoSims = meanProtoSims;
            hvStats(g,p,t).minProtoSims = minProtoSims;
            hvStats(g,p,t).ptMag = ptMag;
            hvStats(g,p,t).uIdx = uIdx;
        end
    end
end

%% gather details on within a gesture
for g = 1:numGest
    allPt = zeros(D,numPos*numTrial);
    labels = cell(1,numPos*numTrial);
    idx = 1;
    for p = 1:numPos
        for t = 1:numTrial
            allPt(:,idx) = hvStats(g,p,t).pt;
            labels{idx} = ['P' num2str(p) '-T' num2str(t)];
            idx = idx + 1;
        end
    end
    
    % prototype similarities
    figure(1)
    set(gcf,'position',[3010 672 1199 891])
    sims = get_cosine_sim(allPt,allPt);
    heatmap(labels,labels,sims,'ColorLimits',[0 1],'FontSize',20,'CellLabelFormat','%0.2g')
    title(['Gesture ' num2str(g) ' prototype similarity'])
end

%% gather details on comparing pairs of vectors
pairs = nchoosek(1:numGest,2);
numPairs = size(pairs,1);
% for i = 1:numPairs
for i = randi(numPairs)
    allPt = zeros(D,2*numPos*numTrial);
    idx = 1;
    for p = 1:numPos
        for t = 1:numTrial
            allPt(:,idx) = hvStats(pairs(i,1),p,t).pt;
            idx = idx + 1;
        end
    end
    for p = 1:numPos
        for t = 1:numTrial
            allPt(:,idx) = hvStats(pairs(i,2),p,t).pt;
            idx = idx + 1;
        end
    end
    
    sims = get_cosine_sim(allPt, allPt);
    imagesc(sims,[0 1])
end

%% useful functions
function [hvOut] = bipolarize_hv(hvIn)
    hvOut = hvIn;
    hvOut(hvOut > 0) = 1;
    hvOut(hvOut < 0) = -1;
    hvOut(hvOut == 0) = (randi(2,length(hvOut(hvOut==0)),1)-1.5)*2;
end

function [sims] = get_cosine_sim(a,b)
%     sims = (a'*b)./(vecnorm(a)'*vecnorm(b));
    D = size(a,1);
    sims = ((a'*b) + D)./2./D;
end