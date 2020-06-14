close all
clear
clc

files1 = dir('./four_trial_adapt*.mat');

testPercentage = 0.05:0.05:0.5;
adaptThreshold = 0.1:0.1:0.8;
% colors = zeros(length(adaptThreshold),3);
% colors(:,3) = (1:length(adaptThreshold))./length(adaptThreshold);
colors = parula(length(adaptThreshold));

accSVM = zeros(length(testPercentage),4);
accHDC = zeros(length(testPercentage),length(adaptThreshold),4);
numSVM = zeros(length(testPercentage),4);
numHDC = zeros(length(testPercentage),length(adaptThreshold),4);

for s = 1:4
    pt1 = load(files1(s).name);
    accSVM(1:10,s) = mean(pt1.accSVM,2);
    numSVM(1:10,s) = mean(pt1.numSVM,2);
    accHDC(1:10,:,s) = mean(pt1.accHDC,3);
    numHDC(1:10,:,s) = mean(pt1.numHDC,3);
end

figure
set(gcf,'position',[500 500 900 400])
subplot(1,2,1)
plot(testPercentage,mean(accSVM,2),':k')
hold on
for i = 1:length(adaptThreshold)
    plot(testPercentage,mean(accHDC(:,i,:),3),'Color',colors(i,:))
end
ylim([0.7 0.95])
xlim([0 1])
grid minor
ylabel('Classification accuracy')
xlabel('Portion of single trial for training')

subplot(1,2,2)
plot(testPercentage,mean(numSVM,2).*320*6/1000,':k')
hold on
for i = 1:length(adaptThreshold)
    plot(testPercentage,mean(numHDC(:,i,:),3).*10,'Color',colors(i,:))
end
ylim([0 1200])
xlim([0 1])
grid minor
ylabel('Memory footprint (k-bits)')
xlabel('Portion of single trial for training')