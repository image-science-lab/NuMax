clear all
close all


addpath(genpath('minFunc'))
addpath('functions');

N = 16*16; %% Size of image. Make sure \sqrt{N} is an interger
siz = 8;%Size of squares. floor(sqrt(N)/2);
delta = 0.4;

%data generation
numSecants = 2500;
secants = zeros(N, numSecants);
[X, Y] = meshgrid(1:9, 1:9);
locStk = [X(:) Y(:)]';
locStk = locStk(:, randperm(size(locStk, 2)));
zz = nchoosek(randperm(size(locStk, 2)), 2)';

%%%Get a random sampling of secants from a translating square manifold
for kk=1:numSecants
    

    for qq=1:2
        tmp = zeros(sqrt(N));
        loc = locStk(:, zz(qq, kk));
        tmp(loc(1)+(0:siz-1), loc(2)+(0:siz-1)) = 1;
        Dpts(:, qq) = tmp(:);
    end
    
    tmp = (Dpts(:,1)-Dpts(:,2));
    secants(:, kk) = tmp/norm(tmp);
end


junk = randperm(numSecants); junk = junk(1:256);
montage(reshape(secants(:, junk), sqrt(N), sqrt(N), 1, []), 'DisplayRange', [min(secants(:)), max(secants(:))]);
title('Sampling of secants')
colormap jet
drawnow


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Parameter setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opt.outer_iterations = 1000;

switch 'l-bfgs'
    case 'grad'
        opt.linear_solver = 'grad';
        opt.tau = 1e-1; % gradient step size
        opt.inner_iterations = 3;
        opt.beta1 = 1e-1; opt.beta2 = 1e-1; %penalty parameters
        opt.eta1 = 1; opt.eta2 = 1; %lagrangian update
    case 'cgs'
        opt.linear_solver = 'cgs';
        opt.linear_iterations = 10;
        opt.inner_iterations = 1;
        opt.beta1 = 1; opt.beta2 = 1; %penalty parameters
        opt.eta1 = 1.618; opt.eta2 = 1.618; %lagrangian update
    case 'l-bfgs'
        opt.linear_solver = 'l-bfgs';
        opt.linear_iterations = 10;
        opt.lbfgs_rank = 5;
        opt.inner_iterations = 1;
        opt.beta1 = 1; opt.beta2 = 1; %penalty parameters
        opt.eta1 = 1.618; opt.eta2 = 1.618; %lagrangian update
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%End parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fA = @(x) funA_secants_WY(x, secants);
fAT = @(x) funAT_secants_WY(x, secants);
[P, r_rank, L_k1, q_k1, Lambda_k1, w_k1] = NuMax(fA, fAT, ones(numSecants, 1), delta, opt);

[U, S, V] = svd(P);
r = rank(P);
U1 = U(:, 1:r);
U1 = (U1 - min(U1(:)))/(max(U1(:))-min(U1(:)));
montage(reshape(U1, sqrt(N), sqrt(N), 1, [])); colormap jet
title('Monatge of the measurement matrix')