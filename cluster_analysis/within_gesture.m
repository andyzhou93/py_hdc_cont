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
numClust = 20;
numHV = 80;

doSave = false;

%% look at stats within a single gesture and arm position
for g = 1:numGest
    
    figure(1);
    set(gcf,'position',[1 1 2048 1184]);
    
    pos = ([]);
    for p = 1:numPos
        res = ([]);
        subplot(4,numPos,p)
        for t = 1:numTrial
            hv = double(experimentData(g,p,t).emgHV(:,experimentData(g,p,t).expGestLabel > 0));
            simPairs = get_cosine_sim(hv,hv) + diag(nan(1,numHV));
            res(t).hv = hv;
            res(t).simPairs = simPairs;
            histEdges = (floor(min(simPairs(:))*100)/100):0.01:(ceil(max(simPairs(:))*100)/100);
            histogram(simPairs(:),histEdges,'Normalization','probability','FaceColor',co(t,:));
            hold on
            xlim([0.45 1])
        end
        hold off
        title(['Position ' num2str(p) ' within trial'])
        ylabel('Probability')
        xlabel('Similarity')
        if p == 1
            legend('T1','T2','T3','Location','northeast')
        end
        
        pairs = nchoosek(1:3,2);
        numPairs = size(pairs,1);
        subplot(4,numPos,p+numPos)
        for i = 1:numPairs
            simPairs = get_cosine_sim(res(pairs(i,1)).hv, res(pairs(i,2)).hv);
            histEdges = (floor(min(simPairs(:))*100)/100):0.01:(ceil(max(simPairs(:))*100)/100);
            histogram(simPairs(:),histEdges,'Normalization','probability','FaceColor',co(i+numTrial,:));
            hold on
            xlim([0.45 1])
        end
        hold off
        title(['Position ' num2str(p) ' across trials'])
        ylabel('Probability')
        xlabel('Similarity')
        if p == 1
            legend(['T' num2str(pairs(1,1)) '-' num2str(pairs(1,2))],['T' num2str(pairs(2,1)) '-' num2str(pairs(2,2))],['T' num2str(pairs(3,1)) '-' num2str(pairs(3,2))],'Location','northeast')
        end
        
        hvAll = zeros(D,numTrial*numHV);
        for t = 1:numTrial
            hvAll(:,(1:numHV)+(t-1)*numHV) = res(t).hv;
        end
        
        pos(p).hvAll = hvAll;
        
        simPairs = get_cosine_sim(hvAll,hvAll) + diag(nan(1,numTrial*numHV));
        histEdges = (floor(min(simPairs(:))*100)/100):0.01:(ceil(max(simPairs(:))*100)/100);
        subplot(4,numPos,p+2*numPos)
        histogram(simPairs(:),histEdges,'Normalization','probability','FaceColor',co(2*numTrial + 1,:));
        xlim([0.45 1])
        title(['Position ' num2str(p) ' all trials'])
        ylabel('Probability')
        xlabel('Similarity')
    end
    
    for p1 = 1:numPos
        subplot(4,numPos,p1+3*numPos)
        for p2 = 1:numPos
            if p1 == p2
                simPairs = get_cosine_sim(pos(p1).hvAll,pos(p2).hvAll) + diag(nan(1,numTrial*numHV));
            else
                simPairs = get_cosine_sim(pos(p1).hvAll,pos(p2).hvAll);
            end
            histEdges = (floor(min(simPairs(:))*100)/100):0.01:(ceil(max(simPairs(:))*100)/100);
            histogram(simPairs(:),histEdges,'Normalization','probability','FaceColor',co(p2 + 2*numTrial + 1,:));
            xlim([0.45 1])
            hold on
            title(['Position ' num2str(p1) ' across positions'])
            ylabel('Probability')
            xlabel('Similarity')
        end
        if p1 == 1
            legend('P1','P2','P3','P4','P5','Location','northeast');
        end
        hold off
    end
    
    sgtitle(['Gesture ' num2str(g)],'FontName','Myriad Pro','FontSize',20)
    if doSave
        saveas(gcf,['./within_gesture_position/G' num2str(g) '.png']);
    end
    drawnow
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