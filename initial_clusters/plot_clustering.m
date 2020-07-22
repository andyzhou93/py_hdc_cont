close all
clear
clc

g1 = 1;
g2 = 2;
p1 = 1;
p2 = p1;

load(['./results_train_within_trial/G' num2str(g1) '_G' num2str(g2) '_P' num2str(p1) '.mat']);

figure(1)
set(gcf,'position',[-134 1418 2048 1184])

for n1 = 1:10
    n2 = n1;
    subplot(2,1,1)
    if n1 == 1
        dendrogram(z1,0,'ColorThreshold',1);
    else
        dendrogram(z1,0,'ColorThreshold',z1(end-n1+2,3));
        hold on
        yline(z1(end-n1+2,3),'k:','LineWidth',2);
        hold off
    end
    xticklabels([]);
    if n1 == 1
        title(['Gesture ' num2str(g1) ' Position ' num2str(p1)])
    else
        title(['Gesture ' num2str(g1) ' Position ' num2str(p1) ' with ' num2str(n1) ' clusters'])
    end

    subplot(2,1,2)
    if n2 == 1
        dendrogram(z2,0,'ColorThreshold',1);
    else
        dendrogram(z2,0,'ColorThreshold',z2(end-n2+2,3));
        hold on
        yline(z2(end-n2+2,3),'k:','LineWidth',2);
        hold off
    end
    xticklabels([]);
    if n2 == 1
        title(['Gesture ' num2str(g2) ' Position ' num2str(p2)])
    else
        title(['Gesture ' num2str(g2) ' Position ' num2str(p2) ' with ' num2str(n2) ' clusters'])
    end
    drawnow
    saveas(gcf,['./cluster_figs/clust' num2str(n1) '.png'])
end