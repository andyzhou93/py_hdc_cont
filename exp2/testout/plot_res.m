close all
clear
clc

figure
set(gcf,'position',[-521 1281 3008 1596])

subplot(11,2,1:2:18)
load('hvRel_none_single_1_cross_10.mat')
heatmap(meanHDAcc,'ColorLimits',[0 1],'FontSize',20,'Colormap',parula)

subplot(11,2,2:2:18)
load('hvRel_none_single_1_within_10.mat')
heatmap(meanHDAcc,'ColorLimits',[0 1],'FontSize',20,'Colormap',parula)

subplot(11,2,[19 21])
load('hvRel_none_single_8_cross_lvqOff_10.mat')
heatmap(meanHDAcc,'ColorLimits',[0 1],'FontSize',20,'Colormap',parula)

subplot(11,2,[20 22])
load('hvRel_none_single_8_within_lvqOff_10.mat')
heatmap(meanHDAcc,'ColorLimits',[0 1],'FontSize',20,'Colormap',parula)
