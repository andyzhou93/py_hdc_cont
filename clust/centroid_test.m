close all
clear
clc

D = 10000;
p1 = ones(D,1);
percDiff = 0.4;
p2 = [-ones(round(percDiff*D),1); ones(round((1 - percDiff)*D),1)];

rad = round(0.4*D);

N = 1000;
v = repmat(p1,1,N);
d1 = zeros(1,N);
d2 = zeros(1,N);
for n = 1:N
%     numChange = round(abs(normrnd(0,1))*rad);
    numChange = rad;
    idx = randperm(D);
    idx = idx(1:numChange);
    v(idx,n) = -v(idx,n);
    d1(n) = sum(v(:,n) == p1);
    d2(n) = sum(v(:,n) == p2);
end

% x = 2:20;
% y = zeros(size(x));
% for i = 1:length(x)
%     y(i) = nchoosek(x(i),2)/factorial(x(i));
% end

mean(d1)

hist(d2)
