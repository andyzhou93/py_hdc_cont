close all
clear
clc

load('emgHV_accel_separate_5_50.mat')

figure
set(gcf,'Position',[1 1 2048 1184])
imagesc(squeeze(prototypeSims)./5, [0 1])
axis square
axis off