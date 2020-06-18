close all
clear
clc

accMax = 1.5;

test = rand(3,3)*2*accMax - accMax;

n = 8;

discBounds = linspace(-accMax,accMax,n+1);

