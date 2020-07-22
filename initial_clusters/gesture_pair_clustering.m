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

% co = colororder;
co = distinguishable_colors(50);
minClust = 3;
maxClust = 40;
numHV = 80;

doSave = false;

%% go through all pairs of gestures
pairs = nchoosek(1:numGest,2);
numPairs = size(pairs,1);

for i = 1:numPairs
    g1 = pairs(i,1);
    g2 = pairs(i,2);
%     g1 = randi(numGest);
%     g2 = randi(numGest);
%     for p1 = 1:numPos
%         for p2 = 1:numPos
    for p1 = 1:numPos
%         for p2 = p1
            p2 = p1;
            hv1 = zeros(D,numTrial*numHV);
            hv2 = zeros(D,numTrial*numHV);
            for t = 1:numTrial
                hv1(:,(1:numHV)+(t-1)*numHV) = double(experimentData(g1,p1,t).emgHV(:,experimentData(g1,p1,t).expGestLabel > 0));
                hv2(:,(1:numHV)+(t-1)*numHV) = double(experimentData(g2,p2,t).emgHV(:,experimentData(g2,p2,t).expGestLabel > 0));
            end
            
%             allHV = [hv1 hv2];
%             label = [ones(1,numTrial*numHV) 2.*ones(1,numTrial*numHV)];
%             
%             simWithin1 = get_cosine_sim(hv1,hv1) + diag(nan(1,numHV*numTrial));
%             simWithin2 = get_cosine_sim(hv2,hv2) + diag(nan(1,numHV*numTrial));
%             simAcross = get_cosine_sim(hv1,hv2);
            
%             figure(1)
%             histogram(simWithin1(:),0:0.01:1,'Normalization','probability');
%             hold on
%             histogram(simWithin2(:),0:0.01:1,'Normalization','probability');
%             histogram(simAcross(:),0:0.01:1,'Normalization','probability');
%             hold off
            
            res = ([]);
            z1 = linkage(hv1','average','hamming');
            z2 = linkage(hv2','average','hamming');
            for n1 = 1:maxClust
%                 for n2 = 1:maxClust
%                 for n2 = n1
                    n2 = n1;
                    disp(['Running G' num2str(g1) 'P' num2str(p1) ' (' num2str(n1) ' clusters) vs. G' num2str(g2) 'P' num2str(p2) ' (' num2str(n2) ' clusters)'])
                    c1 = cluster(z1,'maxclust',n1);
                    c2 = cluster(z2,'maxclust',n2);
                    
%                     figure(1)
%                     set(gcf,'position',[1 1 2048 1184])
%                     
%                     subplot(2,1,1)
%                     if n1 == 1
%                         dendrogram(z1,0,'ColorThreshold',1);
%                     else
%                         dendrogram(z1,0,'ColorThreshold',z1(end-n1+2,3));
%                         hold on
%                         yline(z1(end-n1+2,3),'k:','LineWidth',2);
%                         hold off
%                     end
%                     xticklabels([]);
%                     title(['Gesture ' num2str(g1) ' Position ' num2str(p1) ' with ' num2str(n1) ' clusters'])
%                     
%                     subplot(2,1,2)
%                     if n2 == 1
%                         dendrogram(z2,0,'ColorThreshold',1);
%                     else
%                         dendrogram(z2,0,'ColorThreshold',z2(end-n2+2,3));
%                         hold on
%                         yline(z2(end-n2+2,3),'k:','LineWidth',2);
%                         hold off
%                     end
%                     xticklabels([]);
%                     title(['Gesture ' num2str(g2) ' Position ' num2str(p2) ' with ' num2str(n2) ' clusters'])
%                     drawnow
                    
                    % get gesture 2 clusters first
                    pt2 = zeros(D,n2);
                    for i2 = 1:n2
                        pt2(:,i2) = bipolarize_hv(sum(hv2(:,c2 == i2),2));
                    end
                    
                    % perform loo for gesture 1
                    inStats = zeros(4,numHV*numTrial);
                    outStats = zeros(2,numHV*numTrial);
                    for t1 = 1:numHV*numTrial
                        pt1 = zeros(D,n1);
                        for i1 = 1:n1
                            pt1(:,i1) = bipolarize_hv(sum(hv1(:,(c1 == i1) & ((1:numHV*numTrial)' ~= t1)),2));
                        end
                        sim1 = get_cosine_sim(hv1(:,t1),pt1);
                        inStats(:,t1) = [max(sim1) sim1(c1(t1)) mean(sim1) mean(sim1(setdiff(1:n1,c1(t1))))];
                        sim2 = get_cosine_sim(hv1(:,t1),pt2);
                        outStats(:,t1) = [max(sim2) mean(sim2)];
                    end
                    correct = inStats(1,:) > outStats(1,:);
                    res(n1,n2).inStats1 = mean(inStats,2);
                    res(n1,n2).outStats1 = mean(outStats,2);
                    res(n1,n2).acc1 = sum(correct)/length(correct);
                    
                    % get gesture 1 clusters first
                    pt1 = zeros(D,n1);
                    for i1 = 1:n1
                        pt1(:,i1) = bipolarize_hv(sum(hv1(:,c1 == i1),2));
                    end
                    
                    % perform loo for gesture 2
                    inStats = zeros(4,numHV*numTrial);
                    outStats = zeros(2,numHV*numTrial);
                    for t2 = 1:numHV*numTrial
                        pt2 = zeros(D,n2);
                        for i2 = 1:n2
                            pt2(:,i2) = bipolarize_hv(sum(hv2(:,(c2 == i2) & ((1:numHV*numTrial)' ~= t2)),2));
                        end
                        sim2 = get_cosine_sim(hv2(:,t2),pt2);
                        inStats(:,t2) = [max(sim2) sim2(c2(t2)) mean(sim2) mean(sim2(setdiff(1:n2,c2(t2))))];
                        sim1 = get_cosine_sim(hv2(:,t2),pt1);
                        outStats(:,t2) = [max(sim1) mean(sim1)];
                    end
                    correct = inStats(1,:) > outStats(1,:);
                    res(n1,n2).inStats2 = mean(inStats,2);
                    res(n1,n2).outStats2 = mean(outStats,2);
                    res(n1,n2).acc2 = sum(correct)/length(correct);
                    
                    if (n1 > minClust) && (res(n1,n2).acc1 == 1) && (res(n1,n2).acc2 == 1)
                        break
                    end
                    
%                 end
            end
                
%             save(['./results/G' num2str(g1) '_G' num2str(g2) '_P' num2str(p1)],'res','z1','z2');
                
        
%         end
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