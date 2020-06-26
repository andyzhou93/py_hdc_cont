close all
clear
clc

load('sub1exp0_accHV.mat');

holdFeat = [experimentData(:,1,:).emgFeat];

%%
figure
set(gcf,'position',[1 1 2048 1184])
ax1 = subplot(2,1,1);
histogram(holdFeat,'EdgeColor','none')
hold on
histogram(holdFeat(111:190,:),'EdgeColor','none')
histogram(holdFeat([1:30 271:300],:),'EdgeColor','none')

grid on
grid minor

ax2 = subplot(2,1,2);
histogram(holdFeat,'Normalization','cdf','EdgeColor','none')
hold on
histogram(holdFeat(111:190,:),'Normalization','cdf','EdgeColor','none')
histogram(holdFeat([1:30 271:300],:),'Normalization','cdf','EdgeColor','none')

linkaxes([ax1 ax2],'x')
grid on
grid minor