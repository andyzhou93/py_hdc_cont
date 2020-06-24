close all
clear
clc

load('randIM_hv.mat');

%%
posIM = zeros(10000,5);
% posVals = zeros(3,5);

for p = 1:5
    for g = 1:13
        for t = 1:3
            posIM(:,p) = posIM(:,p) + sum(double(experimentData(g,p,t).accHV64(:,experimentData(g,p,t).expGestLabel > 0)),2);
%             posVals(:,p) = posVals(:,p) + sum(experimentData(g,p,t).accMeanFeat(experimentData(g,p,t).expGestLabel > 0,:))';
        end
    end
end

posIM(posIM >= 0) = 1;
posIM(posIM < 0) = -1;

% posVals = posVals./80/13/3;

get_dist(posIM)
% get_dist(posVals)

function [sims] = get_dist(cim)
    sims = (cim'*cim)./(vecnorm(cim)'*vecnorm(cim));
end