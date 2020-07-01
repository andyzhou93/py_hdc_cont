close all
clear
clc

threshold = [10 30 50 55 60];

figure
set(gcf,'position',[1 1 1000 500])

contextEncode = 'none';

results = zeros(length(threshold),5);
for t = 1:length(threshold)
    filename = ['emgHV_' contextEncode '_auto_' num2str(threshold(t)) '_5_10.mat'];
    if exist(filename,'file')
        res = load(filename);
        results(t,:) = res.meanHDAcc;
    else
        results(t,:) = nan;
    end
end

subplot(1,3,1)
plot(threshold,results,'*-')
hold on
plot(threshold,mean(results,2),'*-k')
ylim([0 1])
xlim([min(threshold) max(threshold)])

contextEncode = 'random';

results = zeros(length(threshold),5);
for t = 1:length(threshold)
    filename = ['emgHV_' contextEncode '_auto_' num2str(threshold(t)) '_5_10.mat'];
    if exist(filename,'file')
        res = load(filename);
        results(t,:) = res.meanHDAcc;
    else
        results(t,:) = nan;
    end
end

subplot(1,3,2)
plot(threshold,results,'*-')
hold on
plot(threshold,mean(results,2),'*-k')
ylim([0 1])
xlim([min(threshold) max(threshold)])

contextEncode = 'accel';

results = zeros(length(threshold),5);
for t = 1:length(threshold)
    filename = ['emgHV_' contextEncode '_auto_' num2str(threshold(t)) '_5_10.mat'];
    if exist(filename,'file')
        res = load(filename);
        results(t,:) = res.meanHDAcc;
    else
        results(t,:) = nan;
    end
end

subplot(1,3,3)
plot(threshold,results,'*-')
hold on
plot(threshold,mean(results,2),'*-k')
ylim([0 1])
xlim([min(threshold) max(threshold)])
