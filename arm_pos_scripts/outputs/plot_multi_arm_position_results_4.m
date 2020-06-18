close all
clear
clc

load('arm_position_4.mat')

figure('Name','HDC accuracy','NumberTitle','off')
set(gcf,'Position',[1349 819 569 436])
plot_grid_results(meanHDAcc.*100);
figure('Name','SVM accuracy','NumberTitle','off')
set(gcf,'Position',[1349 819 569 436])
plot_grid_results(meanSVMAcc.*100);

numCombs = size(clustHits,1);
numPositions = size(clustHits,2);

figure('Name','HDC cluster hits - all','NumberTitle','off')
set(gcf,'position',[1 1173 3008 424])
for i = 1:numCombs
    c = squeeze(clustHits(i,:,:));
    c = c./repmat(sum(c,2),1,numPositions).*100;
    subplot(1,numCombs,i)
    plot_grid_results(c)
end

figure('Name','HDC cluster hits - correct','NumberTitle','off')
set(gcf,'position',[1 1173 3008 424])
for i = 1:numCombs
    c = squeeze(clustCorrectHits(i,:,:));
    c = c./repmat(sum(c,2),1,numPositions).*100;
    subplot(1,numCombs,i)
    plot_grid_results(c)
end

figure('Name','HDC cluster hits - incorrect','NumberTitle','off')
set(gcf,'position',[1 1173 3008 424])
for i = 1:numCombs
    c = squeeze(clustIncorrectHits(i,:,:));
    c = c./repmat(sum(c,2),1,numPositions).*100;
    subplot(1,numCombs,i)
    plot_grid_results(c)
end

function [] = plot_grid_results(acc)
    numCombs = size(acc,1);
    numPositions = size(acc,2);
    
    imagesc(acc,[0 100])
    axis equal
    axis off
    xticks([])
    yticks([])
    xticklabels({})
    yticklabels({})
    
    for i = 1:numCombs
        for j = 1:numPositions
            if acc(i,j) > 50
                text(j,i, num2str(acc(i,j),'%.2f%%'),'Color','black','FontSize',16,'VerticalAlignment','middle','HorizontalAlignment','center')
            else
                text(j,i, num2str(acc(i,j),'%.2f%%'),'Color','white','FontSize',16,'VerticalAlignment','middle','HorizontalAlignment','center')
            end
        end
    end
end