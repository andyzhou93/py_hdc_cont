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
            disp(['Running G' num2str(g1) 'P' num2str(p1) ' vs. G' num2str(g2) 'P' num2str(p2)])
            
            
            %% testing gesture 1 first
            hv1Train = zeros(D,(numTrial-1)*numHV);
            hv1Test = zeros(D,numHV);
            hv2 = zeros(D,numTrial);
            
            % static gesture 2 training set
            idx = 1:numHV;
            for t = 1:numTrial
                hv2(:,idx) = double(experimentData(g2,p2,t).emgHV(:,experimentData(g2,p2,t).expGestLabel > 0));
                idx = idx + numHV;
            end
            z2 = linkage(hv2','average','hamming');
            
            inStats1 = zeros(maxClust,2);
            outStats1 = zeros(maxClust,2);
            acc1 = zeros(maxClust,1);
            
            % loop through testing trials for gesture 1
            for testTrial = 1:numTrial
                idx = 1:numHV;
                for trainTrial = setdiff(1:numTrial,testTrial)
                    hv1Train(:,idx) = double(experimentData(g1,p1,trainTrial).emgHV(:,experimentData(g1,p1,trainTrial).expGestLabel > 0));
                    idx = idx + numHV;
                end
                hv1Test = double(experimentData(g1,p1,testTrial).emgHV(:,experimentData(g1,p1,testTrial).expGestLabel > 0));
                
                z1 = linkage(hv1Train','average','hamming');
                
                for n = 1:maxClust
                    c1 = cluster(z1,'maxclust',n);
                    pt1 = zeros(D,n);
                    for i1 = 1:n
                        pt1(:,i1) = bipolarize_hv(sum(hv1Train(:,c1 == i1),2));
                    end
                    
                    c2 = cluster(z2,'maxclust',n);
                    pt2 = zeros(D,n);
                    for i2 = 1:n
                        pt2(:,i2) = bipolarize_hv(sum(hv2(:,c2 == i2),2));
                    end
                    
                    sim1 = get_cosine_sim(hv1Test,pt1);
                    sim2 = get_cosine_sim(hv1Test,pt2);
                    correct = max(sim1,[],2) > max(sim2,[],2);
                    
                    inStats1(n,:) = inStats1(n,:) + [mean(max(sim1,[],2)), mean(mean(sim1,2))];
                    outStats1(n,:) = outStats1(n,:) + [mean(max(sim2,[],2)), mean(mean(sim2,2))];
                    acc1(n) = acc1(n) + sum(correct)/length(correct);
                end
            end
            
            inStats1 = inStats1./numTrial;
            outStats1 = outStats1./numTrial;
            acc1 = acc1./numTrial;
            
            %% testing gesture 2
            hv2Train = zeros(D,(numTrial-1)*numHV);
            hv2Test = zeros(D,numHV);
            hv1 = zeros(D,numTrial);
            
            % static gesture 1 training set
            idx = 1:numHV;
            for t = 1:numTrial
                hv1(:,idx) = double(experimentData(g1,p1,t).emgHV(:,experimentData(g1,p1,t).expGestLabel > 0));
                idx = idx + numHV;
            end
            z1 = linkage(hv1','average','hamming');
            
            inStats2 = zeros(maxClust,2);
            outStats2 = zeros(maxClust,2);
            acc2 = zeros(maxClust,1);
            
            % loop through testing trials for gesture 2
            for testTrial = 1:numTrial
                idx = 1:numHV;
                for trainTrial = setdiff(1:numTrial,testTrial)
                    hv2Train(:,idx) = double(experimentData(g2,p2,trainTrial).emgHV(:,experimentData(g2,p2,trainTrial).expGestLabel > 0));
                    idx = idx + numHV;
                end
                hv2Test = double(experimentData(g2,p2,testTrial).emgHV(:,experimentData(g2,p2,testTrial).expGestLabel > 0));
                
                z2 = linkage(hv2Train','average','hamming');
                
                for n = 1:maxClust
                    c1 = cluster(z1,'maxclust',n);
                    pt1 = zeros(D,n);
                    for i1 = 1:n
                        pt1(:,i1) = bipolarize_hv(sum(hv1(:,c1 == i1),2));
                    end
                    
                    c2 = cluster(z2,'maxclust',n);
                    pt2 = zeros(D,n);
                    for i2 = 1:n
                        pt2(:,i2) = bipolarize_hv(sum(hv2Train(:,c2 == i2),2));
                    end
                    
                    sim1 = get_cosine_sim(hv2Test,pt1);
                    sim2 = get_cosine_sim(hv2Test,pt2);
                    correct = max(sim2,[],2) > max(sim1,[],2);
                    
                    inStats2(n,:) = inStats2(n,:) + [mean(max(sim2,[],2)), mean(mean(sim2,2))];
                    outStats2(n,:) = outStats2(n,:) + [mean(max(sim1,[],2)), mean(mean(sim1,2))];
                    acc2(n) = acc2(n) + sum(correct)/length(correct);
                end
            end
            
            inStats2 = inStats2./numTrial;
            outStats2 = outStats2./numTrial;
            acc2 = acc2./numTrial;
            
            save(['./results_train_out_trial/G' num2str(g1) '_G' num2str(g2) '_P' num2str(p1)],'acc1','acc2','inStats1','inStats2','outStats1','outStats2','z1','z2');
        
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