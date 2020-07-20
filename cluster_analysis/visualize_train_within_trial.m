close all
clear
clc

resultDir = './results_train_within_trial/';

numPos = 5;
numGest = 13;

out = ([]);

maxError = 0;
for p = 1:numPos
    out(1,p).acc = zeros(numGest);
    out(1,p).clust = zeros(numGest);
    for g1 = 1:(numGest-1)
        for g2 = (g1+1):numGest
            fname = [resultDir 'G' num2str(g1) '_G' num2str(g2) '_P' num2str(p) '.mat'];
            res = load(fname);
            maxClust = length(res.res);
            
            bestAcc1 = 0;
            bestAcc2 = 0;
            for n = 1:maxClust
                if res.res(n,n).acc1 > bestAcc1
                    bestAcc1 = res.res(n,n).acc1;
                    bestClust1 = n;
                end
                if res.res(n,n).acc2 > bestAcc2
                    bestAcc2 = res.res(n,n).acc2;
                    bestClust2 = n;
                end
            end
            
            out(1,p).acc(g1,g2) = 1-bestAcc1;
            out(1,p).acc(g2,g1) = 1-bestAcc2;
            out(1,p).clust(g1,g2) = bestClust1;
            out(1,p).clust(g2,g1) = bestClust2;
        end
    end
    maxError = max([maxError max(out(1,p).acc(:))]);
end

resultDir = './results_train_out_trial/';

for p = 1:numPos
    out(2,p).acc = zeros(numGest);
    out(2,p).clust = zeros(numGest);
    for g1 = 1:(numGest-1)
        for g2 = (g1+1):numGest
            fname = [resultDir 'G' num2str(g1) '_G' num2str(g2) '_P' num2str(p) '.mat'];
            res = load(fname);
            maxClust = length(res.acc1);
            
            bestAcc1 = 0;
            bestAcc2 = 0;
            for n = 1:maxClust
                if res.acc1(n) > bestAcc1
                    bestAcc1 = res.acc1(n);
                    bestClust1 = n;
                end
                if res.acc2(n) > bestAcc2
                    bestAcc2 = res.acc2(n);
                    bestClust2 = n;
                end
            end
            
            out(2,p).acc(g1,g2) = 1-bestAcc1;
            out(2,p).acc(g2,g1) = 1-bestAcc2;
            out(2,p).clust(g1,g2) = bestClust1;
            out(2,p).clust(g2,g1) = bestClust2;
        end
    end
    maxError = max([maxError max(out(2,p).acc(:))]);
end

figure(1);
set(gcf,'position',[1 1 2048 767]);
for p = 1:numPos
    subplot(2,numPos,p);
%     heatmap(out(1,p).acc,'ColorLimits',[0 1]);
    imagesc(out(1,p).acc,[0 maxError]);
    title(['Training within trials - P' num2str(p)])
    xlabel('Paired gesture')
    ylabel('Tested gesture')
    axis square
    colorbar
    subplot(2,numPos,p+numPos);
    imagesc(out(2,p).acc,[0 maxError]);
    title(['Training across trials - P' num2str(p)])
    xlabel('Paired gesture')
    ylabel('Tested gesture')
    axis square
    colorbar
    sgtitle('Classification errors between gesture pairs','FontName','Myriad Pro','FontSize',20)
end

figure(2);
set(gcf,'position',[1 1 2048 767]);
for p = 1:numPos
    subplot(2,numPos,p);
%     heatmap(out(1,p).acc,'ColorLimits',[0 1]);
    imagesc(out(1,p).clust,[0 maxClust]);
    title(['Training within trials - P' num2str(p)])
    xlabel('Paired gesture')
    ylabel('Tested gesture')
    axis square
    colorbar
    subplot(2,numPos,p+numPos);
    imagesc(out(2,p).clust,[0 maxClust]);
    title(['Training across trials - P' num2str(p)])
    xlabel('Paired gesture')
    ylabel('Tested gesture')
    axis square
    colorbar
    sgtitle('Best number of clusters between gesture pairs','FontName','Myriad Pro','FontSize',20)
end
