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

co = colororder;
numClust = 20;

numHV = 80;

for p = 1:numPos
    for g1 = 1:numGest
        % collect info for gesture 1
        hv1 = zeros(D,numTrial*numHV);
        idx = 1:numHV;
        for t = 1:numTrial
            hv1(:,idx) = double(experimentData(g1,p,t).emgHV(:,experimentData(g1,p,t).expGestLabel > 0));
            idx = idx + numHV;
        end
        simPairs1 = get_cosine_sim(hv1,hv1) + diag(nan(1,numTrial*numHV));
        simPairsStats1 = [nanmin(simPairs1(:)) nanmean(simPairs1(:)) nanmedian(simPairs1(:)) nanmax(simPairs1(:))];
        
        for g2 = (g1+1):numGest
            hv2 = zeros(D,numTrial*numHV);
            idx = 1:numHV;
            for t = 1:numTrial
                hv2(:,idx) = double(experimentData(g2,p,t).emgHV(:,experimentData(g2,p,t).expGestLabel > 0));
                idx = idx + numHV;
            end
            simPairs2 = get_cosine_sim(hv2,hv2) + diag(nan(1,numTrial*numHV));
            simPairsStats2 = [nanmin(simPairs2(:)) nanmean(simPairs2(:)) nanmedian(simPairs2(:)) nanmax(simPairs2(:))];
            
            crossPairs = get_cosine_sim(hv1,hv2);
            crossPairsStats = [nanmin(crossPairs(:)) nanmean(crossPairs(:)) nanmedian(crossPairs(:)) nanmax(crossPairs(:))];
            
            figure(1)
            set(gcf,'position',[1 1 2048 1184]);
            subplot(2,3,1)
            histrange = (floor(min([simPairsStats1(1) simPairsStats2(1) crossPairsStats(1)])*100)/100):0.01:(floor(max([simPairsStats1(end) simPairsStats2(end) crossPairsStats(end)])*100)/100);
            histogram(simPairs1(:),histrange,'Normalization','probability');
            hold on
            histogram(simPairs2(:),histrange,'Normalization','probability');
            histogram(crossPairs(:),histrange,'Normalization','probability');
            legend(['Within gesture' num2str(g1)], ['Within gesture ' num2str(g2)], ['Between gestures ' num2str(g1) ' and ' num2str(g2)]);
            hold off
            ylabel('Probability')
            xlabel('Hamming similarity')
            title('Pairwise similarities')
            
            ptMag1 = sum(hv1,2)./numTrial/numHV;
            ptMag2 = sum(hv2,2)./numTrial/numHV;
            
%             figure(2)
            subplot(2,3,2)
            histogram(abs(ptMag1),0:0.05:1,'Normalization','probability');
            hold on
            histogram(abs(ptMag2),0:0.05:1,'Normalization','probability');
            hold off
            ylabel('Probability')
            xlabel('Element-wise majority')
            legend(['Gesture ' num2str(g1)], ['Gesture ' num2str(g2)],'Location','northwest');
            title('Prototype element majorities')
            
            magThresh = 0:0.05:1;
            e1 = zeros(size(magThresh));
            e2 = zeros(size(magThresh));
            eCommon = zeros(size(magThresh));
            eMatch = zeros(size(magThresh));
            for m = 1:length(magThresh)
                idx1 = find(abs(ptMag1) >= magThresh(m));
                idx2 = find(abs(ptMag2) >= magThresh(m));
                e1(m) = length(idx1);
                e2(m) = length(idx2);
                idxCommon = intersect(idx1,idx2);
                eCommon(m) = length(idxCommon);
                eMatch(m) = sum(sign(ptMag1(idxCommon)) == sign(ptMag2(idxCommon)));
            end
            
%             figure(3)
            subplot(2,3,3)
            plot(magThresh,e1./D);
            hold on
            plot(magThresh,e2./D);
            plot(magThresh,eCommon./D);
            plot(magThresh,eMatch./D);
            ylabel('% elements surpassing majority threshold')
            xlabel('Majority threshold')
            legend(['Gesture ' num2str(g1)], ['Gesture ' num2str(g2)], ['Common to gestures ' num2str(g1) ' and ' num2str(g2)], ['Matching between gestures ' num2str(g1) ' and ' num2str(g2)]);
            hold off
            title('Matching prototype elements')
            
            z1 = linkage(hv1','average','cosine');
            z2 = linkage(hv2','average','cosine');
            pt = ([]);
            for n = 1:numClust
                pt(n).c1 = cluster(z1,'maxclust',n);
                pt(n).c2 = cluster(z2,'maxclust',n);
                pt(n).hv = zeros(D,n*2);
                pt(n).member1 = zeros(numHV*numTrial,n*2);
                pt(n).member2 = zeros(numHV*numTrial,n*2);
                for i = 1:n
                    idx1 = find(pt(n).c1 == i);
                    pt(n).hv(:,i) = bipolarize_hv(sum(hv1(:,idx1),2));
                    pt(n).member1(idx1,i) = 1;
                    idx2 = find(pt(n).c2 == i);
                    pt(n).hv(:,i+n) = bipolarize_hv(sum(hv2(:,idx2),2));
                    pt(n).member2(idx2,i+n) = 1;
                end
                
                pt(n).sims1 = get_cosine_sim(hv1,pt(n).hv);
                pt(n).sims2 = get_cosine_sim(hv2,pt(n).hv);
               
%                 figure(2)
%                 subplot(1,2,1)
%                 plot(pt(n).sims1)
%                 ylim([0 1])
%                 subplot(1,2,2)
%                 plot(pt(n).sims2)
%                 ylim([0 1])
                
                [~,l] = max(pt(n).sims1,[],2);
                pt(n).label1Inclusive = floor((l-1)./n)+1;
                [~,l] = max(pt(n).sims2,[],2);
                pt(n).label2Inclusive = floor((l-1)./n)+1;
                
                pt(n).acc1Inclusive = sum(pt(n).label1Inclusive == 1)/numTrial/numHV;
                pt(n).acc2Inclusive = sum(pt(n).label2Inclusive == 2)/numTrial/numHV;
                
                simsEx = pt(n).sims1;
                idx = logical(pt(n).member1);
                simsEx(idx) = nan;
                [~,l] = nanmax(simsEx,[],2);
                pt(n).label1Exclusive = floor((l-1)./n)+1;
                
                simsEx = pt(n).sims2;
                idx = logical(pt(n).member2);
                simsEx(idx) = nan;
                [~,l] = nanmax(simsEx,[],2);
                pt(n).label2Exclusive = floor((l-1)./n)+1;
                
                pt(n).acc1Exclusive = sum(pt(n).label1Exclusive == 1)/numTrial/numHV;
                pt(n).acc2Exclusive = sum(pt(n).label2Exclusive == 2)/numTrial/numHV;
            end
            subplot(2,3,4)
            plot([pt.acc1Inclusive],':','Color',co(1,:))
            hold on
            plot([pt.acc1Exclusive],'Color',co(1,:))
            plot([pt.acc2Inclusive],':','Color',co(2,:))
            plot([pt.acc2Exclusive],'Color',co(2,:))
            xlabel('Clusters per gesture')
            ylabel('Classification accuracy')
            legend(['Gesture ' num2str(g1) ' inclusive'], ['Gesture ' num2str(g1) ' exclusive'], ['Gesture ' num2str(g2) ' inclusive'], ['Gesture ' num2str(g2) ' exclusive'],'Location','southeast')
            hold off
            title('Clustering accuracy')
            
            bar1 = zeros(numClust);
            bar2 = zeros(numClust);
            for c = 1:numClust
                bar1(c,1:c) = sort(histcounts(pt(c).c1),'descend');
                bar2(c,1:c) = sort(histcounts(pt(c).c2),'descend');
            end
            bar1 = bar1./numTrial/numHV;
            bar2 = bar2./numTrial/numHV;
            
            subplot(2,3,5)
            bar(bar1,'stacked')
            ylabel('% examples per cluster')
            xlabel('Number of clusters')
            title('Gesture 1 cluster sizes')
            
            subplot(2,3,6)
            bar(bar2,'stacked')
            ylabel('% examples per cluster')
            xlabel('Number of clusters')
            title('Gesture 2 cluster sizes')
            
            saveas(gcf,['./gesture_pair_figs/P' num2str(p) '_G' num2str(g1) '_G' num2str(g2) '.png']);
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
%     sims = (a'*b)./(vecnorm(a)'*vecnorm(b));
    D = size(a,1);
    sims = ((a'*b) + D)./2./D;
end