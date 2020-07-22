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

numIter = 100;

for g = 1:numGest
    for p = 1:numPos
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
            
            figure(1)
            set(gcf,'position',[100 100 1200 1100])
            heatmap(pairSims,'ColorLimits',[0 1]);
            title(['Gesture ' num2str(g) ' Position ' num2str(p) ' Trial ' num2str(t)])
            
            z = linkage(hv','single','hamming');
            figure(2)
            set(gcf,'position',[600 300 1100 500])
            dendrogram(z)
            
            c = zeros(numHV,numHV);
            for i = 1:numHV
                c(:,i) = cluster(z,'maxclust',i);
            end
            figure(3)
            imagesc(c(:,1:end-1)')
            colormap jet
            
            numStored = (1:2:numHV)';
            simIn = zeros(length(numStored),numIter);
            simOut = zeros(length(numStored),numIter);
            for n = 1:numIter
                idx = randperm(numHV);
                hvOrdered = hv(:,idx);
                pts = cumsum(hvOrdered,2);
                pts = pts(:,1:2:end);
                pts(pts > 0) = 1;
                pts(pts < 0) = -1;
                sim = get_cosine_sim(pts, hvOrdered);
                for s = 1:length(numStored)
                    simIn(s,n) = mean(sim(s,1:numStored(s)));
                    simOut(s,n) = mean(sim(s,(numStored(s)+1):end));
                end
            end
            figure(4)
            plot(numStored,mean(simIn,2));
            hold on
            plot(numStored,mean(simOut,2));
            hold off
            pause
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