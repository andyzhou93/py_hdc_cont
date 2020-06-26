close all
clear
clc

allFiles = dir('./*.mat');
fileNames = {allFiles.name}';

for f = 1:length(fileNames)
    get_figures(fileNames{f});
    close all
end

function [] = get_figures(file)
    res = load(file);
    numCombs = size(res.meanHDAcc,1);
    numPositions = size(res.meanHDAcc,2);

    %% accuracy figure
    figure('Name','Classification accuracy','NumberTitle','off')
    baseSize = 700;
    baseGest = 13;
    scaledHeight = round(1.7*baseSize*numCombs/baseGest);
    scaledWidth = round(1.7*baseSize*numPositions/baseGest);
    set(gcf,'Position',[100 100 scaledWidth scaledHeight])

    imagesc(res.meanHDAcc.*100,[0 100])
    axis equal
    axis off
    xticks([])
    yticks([])
    xticklabels({})
    yticklabels({})

    for i = 1:numCombs
        for j = 1:numPositions
            if res.meanHDAcc(i,j)*100 > 50
%                 text(j,i, num2str(res.meanHDAcc(i,j)*100,'%.2f%%'),'Color','black','FontSize',15,'VerticalAlignment','middle','HorizontalAlignment','center','FontName','MyriadPro')
                text(j,i, num2str(res.meanHDAcc(i,j)*100,'%.2f%%'),'Color','black','FontSize',15,'VerticalAlignment','middle','HorizontalAlignment','center')
            else
%                 text(j,i, num2str(res.meanHDAcc(i,j)*100,'%.2f%%'),'Color','white','FontSize',15,'VerticalAlignment','middle','HorizontalAlignment','center','FontName','MyriadPro')
                text(j,i, num2str(res.meanHDAcc(i,j)*100,'%.2f%%'),'Color','white','FontSize',15,'VerticalAlignment','middle','HorizontalAlignment','center')
            end
        end
    end
    
    print(['./figs/' file(1:end-4) '_acc'],'-dsvg')
    
    figure('Name','Classification accuracy','NumberTitle','off')
    bar100 = 100;
    inAcc = nan(numCombs,numPositions);
    outAcc = nan(numCombs,numPositions);
    
    for i = 1:numCombs
        trainCombs = res.trainCombinations(i,:)+1;
        notTrainCombs = setdiff(1:5,trainCombs);
        inAcc(i,trainCombs) = res.meanHDAcc(i,trainCombs).*100;
        outAcc(i,notTrainCombs) = res.meanHDAcc(i,notTrainCombs).*100;
    end
    
    bar([bar100 nanmean(inAcc(:)) nanmean(outAcc(:)) nanmean(inAcc,1) nanmean(outAcc,1)])
    xticklabels({'100 percent', 'Within position', 'Across position'})
    xtickangle(60)
    
    print(['./figs/' file(1:end-4) '_acc_bar'],'-dsvg')
    
    %% cluster hits figure
    if isfield(res,'clustHits') && (numCombs == 1)
        figure('Name','Cluster hits during inference','NumberTitle','off')
        scaledHeight = 1184;
        scaledWidth = round(2.5*baseSize*numPositions*numCombs/baseGest);
        set(gcf,'Position',[100 100 scaledWidth scaledHeight])

        for comb = 1:numCombs
            c = squeeze(res.clustHits(comb,:,:));
            totalHits = repmat(sum(c,2),1,numPositions);
            c = c./totalHits.*100;
            subplot(3,numCombs,comb)
            imagesc(transpose(c),[0 100])
            axis equal
            axis off
            xticks([])
            yticks([])
            xticklabels({})
            yticklabels({})

            for i = 1:numPositions
                for j = 1:numPositions
                    if c(i,j) > 50
%                         text(i,j, num2str(c(i,j),'%.2f%%'),'Color','black','FontSize',11,'VerticalAlignment','middle','HorizontalAlignment','center','FontName','MyriadPro')
                        text(i,j, num2str(c(i,j),'%.2f%%'),'Color','black','FontSize',11,'VerticalAlignment','middle','HorizontalAlignment','center')
                    else
%                         text(i,j, num2str(c(i,j),'%.2f%%'),'Color','white','FontSize',11,'VerticalAlignment','middle','HorizontalAlignment','center','FontName','MyriadPro')
                        text(i,j, num2str(c(i,j),'%.2f%%'),'Color','white','FontSize',11,'VerticalAlignment','middle','HorizontalAlignment','center')
                    end
                end
            end
        end

        for comb = 1:numCombs
            c = squeeze(res.clustCorrectHits(comb,:,:));
            totalHits = repmat(sum(c,2),1,numPositions);
            c = c./totalHits.*100;
            subplot(3,numCombs,comb+numCombs)
            imagesc(transpose(c),[0 100])
            axis equal
            axis off
            xticks([])
            yticks([])
            xticklabels({})
            yticklabels({})

            for i = 1:numPositions
                for j = 1:numPositions
                    if c(i,j) > 50
%                         text(i,j, num2str(c(i,j),'%.2f%%'),'Color','black','FontSize',11,'VerticalAlignment','middle','HorizontalAlignment','center','FontName','MyriadPro')
                        text(i,j, num2str(c(i,j),'%.2f%%'),'Color','black','FontSize',11,'VerticalAlignment','middle','HorizontalAlignment','center')
                    else
%                         text(i,j, num2str(c(i,j),'%.2f%%'),'Color','white','FontSize',11,'VerticalAlignment','middle','HorizontalAlignment','center','FontName','MyriadPro')
                        text(i,j, num2str(c(i,j),'%.2f%%'),'Color','white','FontSize',11,'VerticalAlignment','middle','HorizontalAlignment','center')
                    end
                end
            end
        end

        for comb = 1:numCombs
            c = squeeze(res.clustIncorrectHits(comb,:,:));
            totalHits = repmat(sum(c,2),1,numPositions);
            c = c./totalHits.*100;
            subplot(3,numCombs,comb+2*numCombs)
            imagesc(transpose(c),[0 100])
            axis equal
            axis off
            xticks([])
            yticks([])
            xticklabels({})
            yticklabels({})

            for i = 1:numPositions
                for j = 1:numPositions
                    if c(i,j) > 50
%                         text(i,j, num2str(c(i,j),'%.2f%%'),'Color','black','FontSize',11,'VerticalAlignment','middle','HorizontalAlignment','center','FontName','MyriadPro')
                        text(i,j, num2str(c(i,j),'%.2f%%'),'Color','black','FontSize',11,'VerticalAlignment','middle','HorizontalAlignment','center')
                    else
%                         text(i,j, num2str(c(i,j),'%.2f%%'),'Color','white','FontSize',11,'VerticalAlignment','middle','HorizontalAlignment','center','FontName','MyriadPro')
                        text(i,j, num2str(c(i,j),'%.2f%%'),'Color','white','FontSize',11,'VerticalAlignment','middle','HorizontalAlignment','center')
                    end
                end
            end
        end
        
        print(['./figs/' file(1:end-4) '_clust'],'-dsvg')

        figure('Name','Cluster hits during inference','NumberTitle','off')

        for comb = 1:numCombs
            c = squeeze(res.clustHits(comb,:,:));
            totalHits = repmat(sum(c,2),1,numPositions);
            c = c./totalHits;
            
            pHit = diag(c);
            pCorrect = res.meanHDAcc';
            
            c = squeeze(res.clustCorrectHits(comb,:,:));
            totalHits = repmat(sum(c,2),1,numPositions);
            c = c./totalHits;
            
            pHitgCorrect = diag(c);
            
            c = squeeze(res.clustIncorrectHits(comb,:,:));
            totalHits = repmat(sum(c,2),1,numPositions);
            c = c./totalHits;
            
            pHitgIncorrect = diag(c);
            
            pCorrectgHit = pHitgCorrect.*pCorrect./pHit;
            pIncorrectgHit = pHitgIncorrect.*(1-pCorrect)./pHit;
            pCorrectgMiss = (1-pHitgCorrect).*pCorrect./(1-pHit);
            
            bar100 = 1;
            bar([bar100 mean(pHit) mean(pCorrect) mean(pHitgCorrect) mean(pHitgIncorrect) mean(pCorrectgHit) mean(pCorrectgMiss)])
            xticklabels({'100 percent', 'P(Hit)', 'P(Correct)', 'P(Hit | Correct)', 'P(Hit | Incorrect)', 'P(Correct | Hit)', 'P(Correct | Miss)'})
            xtickangle(60)
        end
        
        print(['./figs/' file(1:end-4) '_clust_bar'],'-dsvg')
    end
end