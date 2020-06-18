close all
clear
clc

% load up experiment data without accelerometer ngrams
old = load('sub1exp0_accHV.mat').experimentData;
im = load('../im.mat').im;

% new structure that will include accelerometer ngrams
experimentData = old;
[numGest, numPos, numTrial] = size(experimentData);

D = size([old.emgHV],1);

numCIMLevels = [4 16 64];

emgMax = 0.04;

for n = numCIMLevels
    cim = create_cim(n,D);
    for g = 1:numGest
        for p = 1:numPos
            for t = 1:numTrial
                emgFeat = experimentData(g,p,t).emgFeat;
                emgIdx = floor(emgFeat./emgMax*n + 1);
                emgIdx(emgIdx > n) = n;
                emgHV = encode_spatiotemporal(emgIdx,im,cim);
                experimentData(g,p,t).(['emgHV' num2str(n)]) = int8(emgHV);
            end
        end
    end
end

save('sub1exp0_emgCIM','experimentData','-v7.3');

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

function [hv] = encode_spatiotemporal(features,im,cim)
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

