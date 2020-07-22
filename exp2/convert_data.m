close all
clear
clc

rawDir = '~/Research/gui_lite/gui-lite/data/';

subject = 1;
experiment = 2;

% get all files associated with this subject and experiment
files = dir([rawDir 'S' num2str(subject,'%03d') '_E' num2str(experiment,'%03d') '*.mat']);
fileNames = {files.name}';

% get all unique gestures and arm positions (assuming that there is an
% all-to-all mapping)
gestures = zeros(length(fileNames),1);
armPositions = zeros(length(fileNames),1);
for i = 1:length(fileNames)
    gestures(i) = str2double(fileNames{i}(12:14));
    armPositions(i) = str2double(fileNames{i}(17:19));
end

gestures = unique(gestures);
armPositions = unique(armPositions);
experimentData = struct();

for f = 1:length(fileNames)
    disp(['Processing ' num2str(f) ' of ' num2str(length(fileNames))])
    % load the raw data file for processing
    raw = load([rawDir fileNames{f}]);
    
    % clean the streamed data for crc
    clean = raw.raw;
    % take care of situation where first sample (or group of samples) are
    % crc errors
    firstClean = find(raw.crc == 0,1);
    for i = 1:(firstClean-1)
        clean(i,:) = clean(firstClean,:);
    end
    % now take care of the remaining crc errors
    crcIdx = find(raw.crc);
    crcIdx(crcIdx < firstClean) = [];
    for i = 1:length(crcIdx)
        clean(crcIdx,:) = clean(crcIdx-1,:);
    end
    % remove first 10 samples (from old buffer)
    clean = clean(10:end,:);
    dataLen = size(clean,1) - mod(size(clean,1),50);
    clean = clean(1:dataLen,:);
    % grab emg and accelerometer data and scale accordingly
    emg = double(clean(:,1:64)).*(100/2^15) - 50;
    temp = uint16(clean(:,65:67));
    acc = zeros(size(temp));
    for ch = 1:3
        acc(:,ch) = double(typecast(temp(:,ch),'int16'));
    end
    acc = acc./(2^14);
    
    % gather timing information
    timings = 1 + raw.bufferSecs; % begining buffer
    for r = 1:raw.reps
        timings(end+1) = timings(end) + raw.transitionSecs; % arm position transition
        timings(end+1) = timings(end) + raw.transitionSecs; % gesture transition
        timings(end+1) = timings(end) + raw.gestureSecs; % gesture hold
        timings(end+1) = timings(end) + raw.transitionSecs; % gesture transition
        timings(end+1) = timings(end) + raw.transitionSecs; % arm position transition
        if r ~= raw.reps
            timings(end+1) = timings(end) + raw.relaxSecs; % relax between trials
        end
    end
%     timings(end+1) = timings(end) + raw.bufferSecs; % ending buffer
    timings = double(timings.*1000);
    
    % divide into trials
    gestIdx = find(raw.gestureID == gestures);
    posIdx = find(raw.positionID == armPositions);
    for r = double(1:raw.reps)
        trialStart = timings((r-1)*6 + 1) - double(raw.relaxSecs)*1000/2 + 1;
        trialEnd = timings((r-1)*6 + 6) + double(raw.relaxSecs)*1000/2;
        out = convert_trial(emg(trialStart:trialEnd,:), acc(trialStart:trialEnd,:), timings((1:6) + (r-1)*6)-trialStart+1,raw.gestureID, raw.positionID);
        
        for fn = fieldnames(out)'
            experimentData(gestIdx,posIdx,r).(fn{1}) = out.(fn{1});
        end
    end
end

save('data.mat','experimentData','-v7.3');

% save(['./armPosition/sub' num2str(subject) 'exp' num2str(experiment)],'experimentData','-v7.3')

function [out] = convert_trial(emg,acc,timings,gesture,position)
    out.emgRaw = single(emg);
    out.accRaw = single(acc);
    % get features
    out.emgFeat = single(feature_mav(emg));
    out.accFeat = single(feature_mean_acc(acc));
    % encode to ngram
    load('mem.mat');
    out.emgHV = int8(encode_spatiotemporal(out.emgFeat,double(im)));
    out.emgHVCIM = int8(encode_spatiotemporal_cim(out.emgFeat,double(im),double(cimEmg)));
    out.accHV = int8(encode_acc(out.accFeat,double(cimAcc1),double(cimAcc2),double(cimAcc3)));
    
    % experimental label based only on timings
    timings = timings./50;
    out.expGestLabel = zeros(size(out.emgFeat,1),1);
    out.expPosLabel = zeros(size(out.emgFeat,1),1);
    % gesture transitions
    out.expGestLabel((timings(2)+1):timings(3)) = -gesture;
    out.expGestLabel((timings(4)+1):timings(5)) = -gesture;
    % gesture hold
    out.expGestLabel((timings(3)+1):timings(4)) = gesture;
    % arm position transitions
    out.expPosLabel((timings(1)+1):timings(2)) = -position;
    out.expPosLabel((timings(5)+1):timings(6)) = -position;
    % arm position hold
    out.expPosLabel((timings(2)+1):timings(5)) = position;
    
    out.expGestLabel = int16(out.expGestLabel);
    out.expPosLabel = int16(out.expPosLabel);
end

function [hv] = encode_acc(features,cim1,cim2,cim3)
    D = size(cim1,1);
    n = size(cim1,2);
    accMax = 1;
    accCIMIdx = floor(features.*n/(2*accMax)) + (n/2) + 1;
    accCIMIdx(accCIMIdx > n) = n;
    accCIMIdx(accCIMIdx < 1) = 1;
    l = size(accCIMIdx,1);
    hv = zeros(D,l);
    for i = 1:l
        hv(:,i) = cim1(:,accCIMIdx(i,1)).*cim2(:,accCIMIdx(i,2)).*cim3(:,accCIMIdx(i,3));
    end
end

function [features] = feature_mav(raw)
    windowSize = 50;
    dataLen = length(raw);
    numChannels = size(raw,2);
    numWindow = floor(dataLen/windowSize);
    features = zeros(numWindow,numChannels);
    for i = 1:numWindow
        idx = (1:windowSize) + (i-1)*windowSize;
        features(i,:) = mean(abs(detrend(raw(idx,:))));
    end
end

function [features] = feature_mean_acc(raw)
    windowSize = 50;
    dataLen = length(raw);
    numChannels = size(raw,2);
    numWindow = floor(dataLen/windowSize);
    features = zeros(numWindow,numChannels);
    for i = 1:numWindow
        idx = (1:windowSize) + (i-1)*windowSize;
        features(i,:) = mean(raw(idx,:));
    end
end

function [hv] = encode_spatiotemporal(features,im)
    N = 5;
    numFeatures = size(features,1);
    numChannels = size(features,2);
    D = size(im,1);
    % spatial encoding
    spatialHV = zeros(D,numChannels);
    for i = 1:numFeatures
        spatialHV(:,i) = bipolarize_hv(sum(im.*repmat(features(i,:),D,1),2));
    end
    % temporal encoding
    hv = ones(D,numFeatures-N+1);
    for i = 1:(numFeatures-N+1)
        for n = 0:(N-1)
            hv(:,i) = hv(:,i).*circshift(spatialHV(:,i+n),N-1-n);
        end
    end
end

function [hv] = encode_spatiotemporal_cim(features,im,cim)
    N = 5;
    numFeatures = size(features,1);
    D = size(im,1);
    
    emgIdx = floor(features./0.04*64 + 1);
    emgIdx(emgIdx > 64) = 64;
    % spatial encoding
    spatialHV = zeros(D,numFeatures);
    for i = 1:numFeatures
        temp = cim(:,emgIdx(i,:)).*im;
        spatialHV(:,i) = bipolarize_hv(sum(temp,2) + prod(temp,2));
    end
    % temporal encoding
    hv = ones(D,numFeatures-N+1);
    for i = 1:(numFeatures-N+1)
        for n = 0:(N-1)
            hv(:,i) = hv(:,i).*circshift(spatialHV(:,i+n),N-1-n);
        end
    end
end

function [imBi] = bipolarize_hv(im)
    numHV = size(im,2);
    imBi = zeros(size(im));
    for i = 1:numHV
        hv = im(:,i);
        hv(hv>0) = 1;
        hv(hv<0) = -1;
        numzeros = sum(hv==0);
        hv(hv==0) = gen_random_hv(numzeros);
        imBi(:,i) = hv;
    end
end

function [hv] = gen_random_hv(D)
    hv = ones(D,1);
    flip = randperm(D);
    hv(flip(1:round(D/2))) = -1;
end