close all
clear
clc

im = create_im(64,10000);
cimEmg = create_cim(64,10000);
cimAcc1 = create_cim(64,10000);
cimAcc2 = create_cim(64,10000);
cimAcc3 = create_cim(64,10000);

imagesc(get_dist([cimEmg cimAcc1 cimAcc2 cimAcc3 im],[cimEmg cimAcc1 cimAcc2 cimAcc3 im]))

im = int8(im);
cimEmg = int8(cimEmg);
cimAcc1 = int8(cimAcc1);
cimAcc2 = int8(cimAcc2);
cimAcc3 = int8(cimAcc3);

save('mem.mat')

function [sims] = get_dist(a,b)
    sims = (a'*b)./(vecnorm(a)'*vecnorm(b));
end

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

function [im] = create_im(n,D)
    im = ones(D,n);
    for i = 1:n
        flipBits = randperm(D,floor(D/2));
        im(flipBits,i) = -1;
    end
end