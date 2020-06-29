close all
clear
clc

load('sub1exp0_all.mat');

%%
% close all
% [numGest, numPos, numTrial] = size(experimentData);
% featValues = zeros(numGest,numPos,64);
% 
% for g = 1:numGest
%     for p = 1:numPos
%         feat = [];
%         for t = 1:numTrial
%             feat = [feat; experimentData(g,p,t).emgFeatCARZeroed(111:190,:)];
%         end
%         featValues(g,p,:) = mean(feat);
%     end
% end
% 
% maxFeat = squeeze(max(featValues,[],1));
% minFeat = squeeze(min(featValues,[],1));
% rangeFeat = maxFeat - minFeat;
% 
% figure
% set(gcf,'position',[500 400 1200 600])
% bar(maxFeat);
% hold on
% bar(minFeat,'FaceColor',[1 1 1]);

close all
[numGest, numPos, numTrial] = size(experimentData);
featValues = zeros(numGest,64);

for g = 1:numGest
        feat = [];
        for t = 1:numTrial
            feat = [feat; experimentData(g,3,t).emgFeat(111:190,:)];
        end
        featValues(g,:) = mean(feat);
end

maxFeat = squeeze(max(featValues,[],1));
minFeat = squeeze(min(featValues,[],1));
rangeFeat = maxFeat - minFeat;

subplot(1,5,1)
bar(maxFeat./max(maxFeat),'EdgeColor','none');
hold on
bar(minFeat./max(maxFeat),'FaceColor',[1 1 1],'EdgeColor','none');
ylim([0 1.0])

featValues = zeros(numGest,64);

for g = 1:numGest
        feat = [];
        for t = 1:numTrial
            feat = [feat; experimentData(g,3,t).emgFeatNorm(111:190,:)];
        end
        featValues(g,:) = mean(feat);
end

maxFeat = squeeze(max(featValues,[],1));
minFeat = squeeze(min(featValues,[],1));
rangeFeat = maxFeat - minFeat;

subplot(1,5,2)
bar(maxFeat./max(maxFeat),'EdgeColor','none');
hold on
bar(minFeat./max(maxFeat),'FaceColor',[1 1 1],'EdgeColor','none');
ylim([0 1.0])

featValues = zeros(numGest,64);

for g = 1:numGest
        feat = [];
        for t = 1:numTrial
            feat = [feat; experimentData(g,3,t).emgFeatRel(111:190,:)];
        end
        featValues(g,:) = mean(feat);
end

maxFeat = squeeze(max(featValues,[],1));
minFeat = squeeze(min(featValues,[],1));
rangeFeat = maxFeat - minFeat;

subplot(1,5,3)
bar(maxFeat./max(maxFeat),'EdgeColor','none');
hold on
bar(minFeat./max(maxFeat),'FaceColor',[1 1 1],'EdgeColor','none');
ylim([0 1.0])

featValues = zeros(numGest,64);

for g = 1:numGest
        feat = [];
        for t = 1:numTrial
            feat = [feat; experimentData(g,3,t).emgFeatZeroed(111:190,:)];
        end
        featValues(g,:) = mean(feat);
end

maxFeat = squeeze(max(featValues,[],1));
minFeat = squeeze(min(featValues,[],1));
rangeFeat = maxFeat - minFeat;

subplot(1,5,4)
bar(maxFeat./max(maxFeat),'EdgeColor','none');
hold on
bar(minFeat./max(maxFeat),'FaceColor',[1 1 1],'EdgeColor','none');
ylim([0 1.0])

featValues = zeros(numGest,64);

for g = 1:numGest
        feat = [];
        for t = 1:numTrial
            feat = [feat; experimentData(g,3,t).emgFeatCAR(111:190,:)];
        end
        featValues(g,:) = mean(feat);
end

maxFeat = squeeze(max(featValues,[],1));
minFeat = squeeze(min(featValues,[],1));
rangeFeat = maxFeat - minFeat;

subplot(1,5,5)
bar(maxFeat./max(maxFeat),'EdgeColor','none');
hold on
bar(minFeat./max(maxFeat),'FaceColor',[1 1 1],'EdgeColor','none');
ylim([0 1.0])