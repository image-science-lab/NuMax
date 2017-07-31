function [P, r_rank, num_active, duration] = numax_wrapper_function(data, train_label, delta, classify)

addpath(genpath('minFunc'))
addpath('functions');

%N = size(data, 1); %%Make sure \sqrt{N} is an interger

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Parameter setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opt.outermost_iterations = 200;
opt.outer_iterations = 10;
opt.tol = 1e-4;

opt.display = 0;
opt.verbose = 1;

opt.init_num_secants = 2000;
opt.max_cg_secants = 10000;

if (size(data, 2) < 10000)
    opt.num_cg_steps = 1;
    opt.num_cg_cols = size(data, 2);
else
    opt.num_cg_cols = 10000;
    opt.num_cg_steps = 2*ceil(size(data, 2)/10000);
end

switch 'cgs'
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


ticID = tic;
if (classify)
    [P, r_rank, num_active, secMinMax] = NuMax_CG_Directional(data, train_label, delta, opt);
else
    [P, r_rank, num_active, secMinMax] = NuMax_CG(data, delta, opt);
end
duration = toc(ticID);


return
% 
% [U, S, V] = svd(P);
% r = rank(P);
% U1 = U(:, 1:r);
% U1 = (U1 - min(U1(:)))/(max(U1(:))-min(U1(:)));
% figure(2)
% montage(reshape(U1, sqrt(N), sqrt(N), 1, [])); colormap jet
% title('Montage of masurement matrix')
% 
% %%Some comparisons with other matrices
% [Upca, Spca, Vpca] = svds(data, r);
% 
% Phi_NuMax = (U(:, 1:r)*(S(1:r, 1:r).^(1/2)))';
% Phi_randn = randn(r, N)/sqrt(r);
% Phi_pca = (Upca)'; %*Spca.^(1/2))';
% 
% [dmin_numax, dmax_numax] = get_rip_constants(data(:, randperm(size(data,2), 10000)), Phi_NuMax);
% [dmin_randn, dmax_randn] = get_rip_constants(data(:, randperm(size(data,2), 10000)), Phi_randn);
% [dmin_pca, dmax_pca] = get_rip_constants(data(:, randperm(size(data,2), 10000)), Phi_pca);
% 
% fprintf('Rank of solution: %d\n', r);
% fprintf('Number of active constraints: %d\n', num_active);
% 
% fprintf('RIP constants for.');
% fprintf('NuMax:   min %2.4f  max: %2.4f\n', dmin_numax, dmax_numax);
% fprintf('Randn:   min %2.4f  max: %2.4f\n', dmin_randn, dmax_randn);
% fprintf('PCA:     min %2.4f  max: %2.4f\n', dmin_pca, dmax_pca);
% 
