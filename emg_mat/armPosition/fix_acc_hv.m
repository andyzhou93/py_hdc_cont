close all
clear
clc

load('sub1exp0_accHV.mat')

accMax = 1.5;

n = 64;
D = 10000;
numGest = 13;
numPos = 5;
numTrial = 3;

cim1 = create_cim(n,D);
cim2 = create_cim(n,D);
cim3 = create_cim(n,D);
for g = 1:numGest
    for p = 1:numPos
        for t = 1:numTrial
            accFeat = experimentData(g,p,t).accMeanFeat;
            accCIMIdx = floor(accFeat.*n/(2*accMax)) + (n/2) + 1;
            l = size(accCIMIdx,1);
            accHV = zeros(D,l);
            for i = 1:l
                accHV(:,i) = cim1(:,accCIMIdx(i,1)).*cim2(:,accCIMIdx(i,2)).*cim3(:,accCIMIdx(i,3));
            end
            experimentData(g,p,t).(['accHV' num2str(n)]) = int8(accHV);
        end
    end
end

ed = experimentData;
clearvars experimentData

load('randIM_hv.mat')
for g = 1:numGest
    for p = 1:numPos
        for t = 1:numTrial
            experimentData(g,p,t).accHV64 = ed(g,p,t).accHV64;
        end
    end
end
save('randIM_hv.mat','experimentData','-v7.3')



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
