close all
clear
clc

D = 10000;
circumference = 16;

cim = create_circ_im(circumference,D);

% imagesc(get_dist(cim))

im = zeros(D,64);
flipBits = randperm(D,round(D/2));
flipAmounts = round(linspace(0,round(D/2),4));

for i = 1:32
    im(:,i) = cim(:,floor((i-1)/4)+1);
    im(flipBits(1:flipAmounts(mod(i-1,4)+1)),i) = im(flipBits(1:flipAmounts(mod(i-1,4)+1)),i).*-1;
end

for i = 33:64
    im(:,i) = cim(:,floor((i-1)/4)+1);
    im(flipBits(1:flipAmounts(4-mod(i-1,4))),i) = im(flipBits(1:flipAmounts(4-mod(i-1,4))),i).*-1;
end

% imagesc(get_dist(im))

save('circ_im','im')


    

function [cim] = create_circ_im(n,D)
    cim = ones(D,n);
    flipBits = randperm(D,floor(D/2));
    cim(flipBits,:) = -1;
    flipBits = randperm(D);
    flipAmounts = round(linspace(0,D,n/2 + 1));
    for i = 1:n/2
        cim(flipBits(1:flipAmounts(i)),i) = cim(flipBits(1:flipAmounts(i)),i)*-1;
    end
    for i = (n/2+1):n
        cim(:,i) = cim(:,i-(n/2)).*-1;
    end
end

function [hv] = gen_random_hv(D)
    hv = ones(D,1);
    flip = randperm(D);
    hv(flip(1:round(D/2))) = -1;
end

function [sims] = get_dist(cim)
    sims = (cim'*cim)./(vecnorm(cim)'*vecnorm(cim));
end
