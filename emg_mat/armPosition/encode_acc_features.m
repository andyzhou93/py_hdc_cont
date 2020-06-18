close all
clear
clc

% load up experiment data without accelerometer ngrams
old = load('sub1exp0.mat').experimentData;

% new structure that will include accelerometer ngrams
experimentData = old;
[numGest, numPos, numTrial] = size(experimentData);

D = size([old.emgHV],1);

numCIMLevels = 2.^(2:7);

accMax = 1.5;

for n = numCIMLevels
    cim = create_cim(n,D);
    for g = 1:numGest
        for p = 1:numPos
            for t = 1:numTrial
                accFeat = experimentData(g,p,t).accMeanFeat;
                accCIMIdx = floor(accFeat.*n/(2*accMax)) + (n/2) + 1;
                l = size(accCIMIdx,1);
                accHV = zeros(D,l);
                for i = 1:l
                    accHV(:,i) = cim(:,accCIMIdx(i,1)).*cim(:,accCIMIdx(i,2)).*cim(:,accCIMIdx(i,3));
                end
                experimentData(g,p,t).(['accHV' num2str(n)]) = int8(accHV);
            end
        end
    end
end

save('sub1exp0_accHV','experimentData','-v7.3');

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

