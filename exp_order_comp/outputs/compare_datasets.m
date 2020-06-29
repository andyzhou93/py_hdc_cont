close all
clear
clc

%% exp 0 single
load('./sub1exp0_emgHV_none_single_1_50.mat')

numCombs = size(trainCombinations,1);
numPositions = size(meanHDAcc,2);

figure(1)
imagesc(meanHDAcc,[0 1])
axis equal
axis off
xticks([])
yticks([])
xticklabels({})
yticklabels({})

for i = 1:numCombs
    for j = 1:numPositions
        if meanHDAcc(i,j)*100 > 50
            text(j,i, num2str(meanHDAcc(i,j)*100,'%.2f%%'),'Color','black','FontSize',15,'VerticalAlignment','middle','HorizontalAlignment','center')
        else
            text(j,i, num2str(meanHDAcc(i,j)*100,'%.2f%%'),'Color','white','FontSize',15,'VerticalAlignment','middle','HorizontalAlignment','center')
        end
    end
end

%% exp 1 single
load('./sub1exp1_emgHV_none_single_1_50.mat')

numCombs = size(trainCombinations,1);
numPositions = size(meanHDAcc,2);

figure(2)
imagesc(meanHDAcc,[0 1])
axis equal
axis off
xticks([])
yticks([])
xticklabels({})
yticklabels({})

for i = 1:numCombs
    for j = 1:numPositions
        if meanHDAcc(i,j)*100 > 50
            text(j,i, num2str(meanHDAcc(i,j)*100,'%.2f%%'),'Color','black','FontSize',15,'VerticalAlignment','middle','HorizontalAlignment','center')
        else
            text(j,i, num2str(meanHDAcc(i,j)*100,'%.2f%%'),'Color','white','FontSize',15,'VerticalAlignment','middle','HorizontalAlignment','center')
        end
    end
end

%% exp 0 combined
load('./sub1exp0_emgHV_none_single_5_50.mat')

numCombs = size(trainCombinations,1);
numPositions = size(meanHDAcc,2);

figure(3)
subplot(2,1,1)
imagesc(meanHDAcc,[0 1])
axis equal
axis off
xticks([])
yticks([])
xticklabels({})
yticklabels({})

for i = 1:numCombs
    for j = 1:numPositions
        if meanHDAcc(i,j)*100 > 50
            text(j,i, num2str(meanHDAcc(i,j)*100,'%.2f%%'),'Color','black','FontSize',15,'VerticalAlignment','middle','HorizontalAlignment','center')
        else
            text(j,i, num2str(meanHDAcc(i,j)*100,'%.2f%%'),'Color','white','FontSize',15,'VerticalAlignment','middle','HorizontalAlignment','center')
        end
    end
end

load('./sub1exp0_emgHV_none_separate_5_50.mat')

numCombs = size(trainCombinations,1);
numPositions = size(meanHDAcc,2);

figure(3)
subplot(2,1,2)
imagesc(meanHDAcc,[0 1])
axis equal
axis off
xticks([])
yticks([])
xticklabels({})
yticklabels({})

for i = 1:numCombs
    for j = 1:numPositions
        if meanHDAcc(i,j)*100 > 50
            text(j,i, num2str(meanHDAcc(i,j)*100,'%.2f%%'),'Color','black','FontSize',15,'VerticalAlignment','middle','HorizontalAlignment','center')
        else
            text(j,i, num2str(meanHDAcc(i,j)*100,'%.2f%%'),'Color','white','FontSize',15,'VerticalAlignment','middle','HorizontalAlignment','center')
        end
    end
end

%% exp 1 combined
load('./sub1exp1_emgHV_none_single_5_50.mat')

numCombs = size(trainCombinations,1);
numPositions = size(meanHDAcc,2);

figure(4)
subplot(2,1,1)
imagesc(meanHDAcc,[0 1])
axis equal
axis off
xticks([])
yticks([])
xticklabels({})
yticklabels({})

for i = 1:numCombs
    for j = 1:numPositions
        if meanHDAcc(i,j)*100 > 50
            text(j,i, num2str(meanHDAcc(i,j)*100,'%.2f%%'),'Color','black','FontSize',15,'VerticalAlignment','middle','HorizontalAlignment','center')
        else
            text(j,i, num2str(meanHDAcc(i,j)*100,'%.2f%%'),'Color','white','FontSize',15,'VerticalAlignment','middle','HorizontalAlignment','center')
        end
    end
end

load('./sub1exp1_emgHV_none_separate_5_50.mat')

numCombs = size(trainCombinations,1);
numPositions = size(meanHDAcc,2);

figure(4)
subplot(2,1,2)
imagesc(meanHDAcc,[0 1])
axis equal
axis off
xticks([])
yticks([])
xticklabels({})
yticklabels({})

for i = 1:numCombs
    for j = 1:numPositions
        if meanHDAcc(i,j)*100 > 50
            text(j,i, num2str(meanHDAcc(i,j)*100,'%.2f%%'),'Color','black','FontSize',15,'VerticalAlignment','middle','HorizontalAlignment','center')
        else
            text(j,i, num2str(meanHDAcc(i,j)*100,'%.2f%%'),'Color','white','FontSize',15,'VerticalAlignment','middle','HorizontalAlignment','center')
        end
    end
end