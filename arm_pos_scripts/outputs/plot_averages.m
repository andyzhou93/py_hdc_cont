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
    bar100 = 100;
    inAcc = 0;
    outAcc = 0;
    for i = 1:numCombs
        trainCombs = res.trainCombinations(i,:)+1;
        notTrainCombs = setdiff(1:5,trainCombs);
        inAcc = inAcc + mean(res.meanHDAcc(i,trainCombs))/numCombs*100;
        outAcc = outAcc + mean(res.meanHDAcc(i,notTrainCombs))/numCombs*100;
    end
    
    bar([bar100 inAcc outAcc])
    xticklabels({'100 percent', 'Within position', 'Across position'})
    xtickangle(60)
      
    print(['./figs/' file(1:end-4) '_acc_avg'],'-dsvg')
    
    %% cluster hits figure
    if isfield(res,'clustHits') && (numCombs == 1)
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

        print(['./figs/' file(1:end-4) '_clust_avg'],'-dsvg')
        
    end
end