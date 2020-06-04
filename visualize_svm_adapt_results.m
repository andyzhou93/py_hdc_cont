close all
clear
clc

load('s2_adaptive_test.mat')

tp = 0.05:0.05:1;

blues = zeros(16,3);
blues(:,3) = (1:16)./16;

accSVM = accSVM(2,:);
numSVM = numSVM(2,:);

accHDC = squeeze(accHDC(2,:,:));
numHDC = squeeze(numHDC(2,:,:));

figure
set(gcf,'Position',[1000 800 1000 1000])
subplot(2,1,1)
for i = 1:16
    plot(tp,accHDC(:,i),'Color',blues(i,:))
    hold on
end
plot(tp,accSVM,'r')
ylabel('Accuracy')
xlabel('Percentage of a trial used for training')
grid on

subplot(2,1,2)
for i = 1:16
    plot(tp,numHDC(:,i)*10000,'Color',blues(i,:))
    hold on
end
plot(tp,numSVM.*320*6,'r')
grid on
ylabel('Memory requirement (bits)')
xlabel('Percentage of a trial used for training')