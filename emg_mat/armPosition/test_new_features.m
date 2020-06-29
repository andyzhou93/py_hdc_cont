close all
clear
clc

%% load up experiment data 
load('sub1exp0.mat','experimentData');
[numGest, numPos, numTrial] = size(experimentData);
D = size([experimentData.emgHV],1);
cim = create_cim(64,D);
cim1 = create_cim(64,D);
cim2 = create_cim(64,D);
cim3 = create_cim(64,D);
im = load('../im.mat').im;

%% get normalized and relative features and encoded ngrams
for p = 1:numPos
    % get the baseline values of rest during each arm position
    featBaseline = [];
    for t = 1:numTrial
        featBaseline = [featBaseline; experimentData(1,p,t).emgFeat(experimentData(1,p,t).expGestLabel > 0,:)];
    end
    featBaselineAvg = mean(featBaseline); 
    
    % calculate the normalized, relative, and zeroed features based on baseline
    for g = 1:numGest
        for t = 1:numTrial
            % reconvert regular features 
            experimentData(g,p,t).emgHV = int8(encode_spatiotemporal(experimentData(g,p,t).emgFeat,im));
            emgIdx = floor(experimentData(g,p,t).emgFeat./0.04*64 + 1);
            emgIdx(emgIdx > 64) = 64;
            experimentData(g,p,t).emgHV64 = encode_spatiotemporal_cim(emgIdx,im,cim);
            % normalized
            experimentData(g,p,t).emgFeatNorm = experimentData(g,p,t).emgFeat./featBaselineAvg;
            experimentData(g,p,t).emgHVNorm = int8(encode_spatiotemporal(experimentData(g,p,t).emgFeatNorm,im));
            % relative
            experimentData(g,p,t).emgFeatRel = experimentData(g,p,t).emgFeat-featBaselineAvg;
            experimentData(g,p,t).emgHVRel = int8(encode_spatiotemporal(experimentData(g,p,t).emgFeatRel,im));
            % zeroed
            experimentData(g,p,t).emgFeatZeroed = experimentData(g,p,t).emgFeat - min(experimentData(g,p,t).emgFeat,[],2);
            experimentData(g,p,t).emgHVZeroed = int8(encode_spatiotemporal(experimentData(g,p,t).emgFeatZeroed,im));
            
            % get common average referenced feature
            experimentData(g,p,t).emgFeatCAR = feature_mav_car(experimentData(g,p,t).emgRaw);
            experimentData(g,p,t).emgHVCAR = int8(encode_spatiotemporal(experimentData(g,p,t).emgFeatCAR,im));
        end
    end
    
    % get the baseline values of rest during each arm position
    featBaseline = [];
    for t = 1:numTrial
        featBaseline = [featBaseline; experimentData(1,p,t).emgFeatCAR(experimentData(1,p,t).expGestLabel > 0,:)];
    end
    featBaselineAvg = mean(featBaseline); 
    
    % calculate the normalized, relative, and zeroed CAR features
    for g = 1:numGest
        for t = 1:numTrial
            % normalized
            experimentData(g,p,t).emgFeatCARNorm = experimentData(g,p,t).emgFeatCAR./featBaselineAvg;
            experimentData(g,p,t).emgHVCARNorm = int8(encode_spatiotemporal(experimentData(g,p,t).emgFeatCARNorm,im));
            % relative
            experimentData(g,p,t).emgFeatCARRel = experimentData(g,p,t).emgFeatCAR-featBaselineAvg;
            experimentData(g,p,t).emgHVCARRel = int8(encode_spatiotemporal(experimentData(g,p,t).emgFeatCARRel,im));
            % zeroed
            experimentData(g,p,t).emgFeatCARZeroed = experimentData(g,p,t).emgFeatCAR - min(experimentData(g,p,t).emgFeatCAR,[],2);
            experimentData(g,p,t).emgHVCARZeroed = int8(encode_spatiotemporal(experimentData(g,p,t).emgFeatCARZeroed,im));
        end
    end
    accMax = 1.5;
    for g = 1:numGest
        for t = 1:numTrial
            accFeat = experimentData(g,p,t).accMeanFeat;
            accCIMIdx = floor(accFeat.*64/(2*accMax)) + (64/2) + 1;
            l = size(accCIMIdx,1);
            accHV = zeros(D,l);
            for i = 1:l
                accHV(:,i) = cim1(:,accCIMIdx(i,1)).*cim2(:,accCIMIdx(i,2)).*cim3(:,accCIMIdx(i,3));
            end
            experimentData(g,p,t).(['accHV64']) = int8(accHV);
        end
    end
    
end

save('sub1exp0_all','experimentData','-v7.3')

% experimentData = rmfield(experimentData,{'emgRaw','accRaw','emgFeat','accMeanFeat','accStdFeat','emgFeatNorm','emgFeatRel','emgFeatZeroed','emgFeatCAR','emgFeatCARNorm','emgFeatCARRel','emgFeatCARZeroed'});
% save('sub1exp1_hv','experimentData','-v7.3')

%% useful functions
function [cim] = create_cim(n,D)
    cim = ones(D,n);
    flipBits = randperm(D,floor(D/2));
    cim(flipBits,:) = -1;
    flipBits = randperm(D,floor(D/2));
    flipAmounts = round(linspace(0,floor(D/2),n));
    for i = 1:n
        cim(flipBits(1:flipAmounts(i)),i) = cim(flipBits(1:flipAmounts(i)),i)*-1;
    end
end

function [sims] = get_dist(cim)
    sims = (cim'*cim)./(vecnorm(cim)'*vecnorm(cim));
end

function [hv] = encode_spatiotemporal_cim(features,im,cim)
    N = 5;
    numFeatures = size(features,1);
    D = size(im,1);
    % spatial encoding
    spatialHV = zeros(D,numFeatures);
    for i = 1:numFeatures
        temp = cim(:,features(i,:)).*im;
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

function [features] = feature_mav_car(raw)
    windowSize = 50;
    dataLen = length(raw);
    numChannels = size(raw,2);
    numWindow = floor(dataLen/windowSize);
    features = zeros(numWindow,numChannels);
    for i = 1:numWindow
        idx = (1:windowSize) + (i-1)*windowSize;
        seg = detrend(raw(idx,:));
        seg = seg - mean(seg,2);
        features(i,:) = mean(abs(seg));
    end
end