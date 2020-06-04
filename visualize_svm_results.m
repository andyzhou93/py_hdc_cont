close all
clear
clc

load('HDCvsSVM.mat')

tp = 0.05:0.05:1;

figure
set(gcf,'Position',[1000 800 1000 1000])
subplot(2,1,1)
plot(tp,mean(mean(accHDC,1),3))
hold on
plot(tp,mean(mean(accSVM,1),3))
ylabel('Accuracy')
xlabel('Percentage of a trial used for training')
grid on

subplot(2,1,2)
plot(tp,ones(size(tp)).*10000*26)
hold on
plot(tp,mean(mean(numSVM,1),3).*320*6)
grid on
ylabel('Memory requirement (bits)')
xlabel('Percentage of a trial used for training')
