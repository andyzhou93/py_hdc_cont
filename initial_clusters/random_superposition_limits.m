close all
clear
clc

D = 10000;
seed = gen_rand_hv(D);

numVectors = 10000;
numIter = 5;
numStored = (1:2:numVectors)';

% percentFreezeBits = 0:0.1:1;
% percentFlipBits = 0:0.05:1;

percentFreezeBits = 0.0;
percentFlipBits = 0.1;

res = ([]);
for i = 1:length(percentFreezeBits)
    for j = 1:length(percentFlipBits)
        if percentFreezeBits(i) + percentFlipBits(j) <= 1
            numFreezeBits = round(percentFreezeBits(i)*D);
            freezeBits = randperm(D,numFreezeBits);
            flippableBits = setdiff(1:D,freezeBits);

            numFlipBits = round(percentFlipBits(j)*D);

            hv = zeros(D,numVectors);
            for n = 1:numVectors
                newHv = seed;
                flipBits = datasample(flippableBits,numFlipBits,'Replace',false);
                newHv(flipBits) = -newHv(flipBits);
                hv(:,n) = newHv;
            end

            s = get_cosine_sim(hv,hv);
            s = s + diag(nan(1,numVectors));
            res(i,j).sims = s(~isnan(s));
            res(i,j).simMean = mean(res(i,j).sims);
            res(i,j).simStd = std(res(i,j).sims);
            simIn = zeros(length(numStored),numIter);
            simOut = zeros(length(numStored),numIter);
            for n = 1:numIter
                idx = randperm(numVectors);
                hvOrdered = hv(:,idx);
                pts = cumsum(hvOrdered,2);
                pts = pts(:,1:2:end);
                pts(pts > 0) = 1;
                pts(pts < 0) = -1;
                sim = get_cosine_sim(pts, hvOrdered);
                for s = 1:length(numStored)
                    simIn(s,n) = mean(sim(s,1:numStored(s)));
                    simOut(s,n) = mean(sim(s,(numStored(s)+1):end));
                end
            end
            res(i,j).simIn = simIn;
            res(i,j).simOut = simOut;
            
            figure(1)
            set(gcf,'Position',[400 400 1000 400])
            plot(numStored,mean(res(i,j).simIn,2))
            hold on
            plot(numStored,mean(res(i,j).simOut,2))
            xlabel('Number of examples superimposed')
            ylabel('Hamming distance')
            legend('Constituents','Remaining examples')
            title([num2str(numFreezeBits) ' frozen bits, ' num2str(numFlipBits) ' randomly flipped bits, average Hamming distance ' num2str(res(i,j).simMean)])
        end
    end
end

function [hv] = gen_rand_hv(D)
    idx = randperm(D,round(D/2));
    hv = ones(D,1);
    hv(idx) = -1;
end

function [sims] = get_cosine_sim(a,b)
%     sims = (a'*b)./(vecnorm(a)'*vecnorm(b));
    D = size(a,1);
    sims = 1 - ((a'*b) + D)./2./D;
end
