close all
clear
clc

rawDir = '~/Research/gui_lite/gui-lite/data/';

subject = 1;
experiment = 1;

% get all files associated with this subject and experiment
files = dir([rawDir 'S' num2str(subject,'%03d') '_E' num2str(experiment,'%03d') '*.mat']);
fileNames = {files.name};

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

for i = 1:length(fileNames)
    raw = load([rawDir fileNames{i}]);
end