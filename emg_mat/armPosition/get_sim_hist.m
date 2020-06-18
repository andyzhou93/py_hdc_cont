close all
clear
clc

load('sub1exp0_emgCIM.mat')
[numGest, numPos, numTrial] = size(experimentData);

histEdges = 0:0.05:1;
allCounts = zeros(1,length(histEdges)-1);
for g = 1:numGest
    for p = 1:numPos
        for t = 1:numTrial
            hv = experimentData(g,p,t).emgHV(:,111:190);
            allCounts = allCounts + histcounts(get_dist(double(hv)),histEdges);
        end
    end
end
figure
plot(histEdges(1:end-1),allCounts)
hold on

allCounts = zeros(1,length(histEdges)-1);
for g = 1:numGest
    for p = 1:numPos
        for t = 1:numTrial
            hv = experimentData(g,p,t).emgHV4(:,111:190);
            allCounts = allCounts + histcounts(get_dist(double(hv)),histEdges);
        end
    end
end
plot(histEdges(1:end-1),allCounts)

allCounts = zeros(1,length(histEdges)-1);
for g = 1:numGest
    for p = 1:numPos
        for t = 1:numTrial
            hv = experimentData(g,p,t).emgHV16(:,111:190);
            allCounts = allCounts + histcounts(get_dist(double(hv)),histEdges);
        end
    end
end
plot(histEdges(1:end-1),allCounts)

allCounts = zeros(1,length(histEdges)-1);
for g = 1:numGest
    for p = 1:numPos
        for t = 1:numTrial
            hv = experimentData(g,p,t).emgHV64(:,111:190);
            allCounts = allCounts + histcounts(get_dist(double(hv)),histEdges);
        end
    end
end
plot(histEdges(1:end-1),allCounts)

legend('Weighted sum','CIM-4','CIM-16','CIM-64')

function [sims] = get_dist(cim)
    sims = (cim'*cim)./(vecnorm(cim)'*vecnorm(cim));
end
