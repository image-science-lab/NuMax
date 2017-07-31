function [P_k1, r_rank, M, secMinMax, L_k1, q_k1, Lambda_k1, w_k1] = NuMax_CG(data, delta, opt)
%%%%%%%%%%%%%%
% Computes low-dimensional isometric linear embedding of a data matrix
%
%  Solves the following optimization problem
%  P_k1 = arg max trace(P)   s.t   1-delta <= v_ij^T P v_ij  <= 1+delta
%
%               where v_ij = data(:, i)-data(:,j)/ ||data(:,i)-data(:,j)||
%
%  The code uses a variant of column generation to do this
% INPUT
% data - N x T matrix of T samples; each sample is a point in R^N
% opt - optimization parameters (see below)
%
% OUTPUT
% P_k1 - Main Solution. P_k1 is a NxN symmetric matrix.
%        P_k1 is expected to be low rank
%        If [U,S,V] = svd(P_k1), then embedding is given by
%          sqrt(S(1:r, 1:r))*U(:, 1:r)'
%          where r is the rank of P_k1
% r_rank - Rank of the solution
% M - # of active constraints at solution
% secMinMax - Violation of feasility condition for each CG round
%
% Other outputs are not necessary
%
% PARAMETERS
%   Look at NuMax.m
%   In addition to that, we need to define
%      opt.outermost_iterations: outermost iterations
%      opt.init_num_secants: Initial # of secants to be used
%      opt.num_cg_steps: # of CG steps
%      opt.num_cg_cols: # of secants to generate in each CG step
%      opt.max_cg_secants: Max # of secants to add in each CG step
if ~exist('opt')
    opt = [];
end
opt = getDefaultParameters(opt);
if ~isfield(opt, 'init_num_secants')
    opt.init_num_secants = size(data, 2);
end
if ~(isfield(opt, 'num_cg_steps') && isfield(opt, 'num_cg_cols'))
    if (size(data, 2) < 5000)
        opt.num_cg_cols = size(data, 2); 
        opt.num_cg_steps = 1;
    else
        opt.num_cg_steps = 2*ceil(size(data, 2)/5000);
        opt.num_cg_cols = 5000;
    end
end
if ~isfield(opt, 'max_cg_secants')
    opt.max_cg_secants = 10000;
end

N = size(data, 1);
T = size(data, 2);

secMinMax = []; idxStk = [];

%just select some secants for initalization
idx1 = random('unid',size(data, 2), opt.init_num_secants, 1);
idx2 = random('unid',size(data, 2), opt.init_num_secants, 1);
throw_idx = find(idx1 == idx2);
idx1(throw_idx) = []; idx2(throw_idx) = [];
idxStk = [idxStk; idx1(:) idx2(:)];
idxStk = sort(idxStk, 2);

secants = data(:, idx1)-data(:, idx2);
secants = secants*diag(1./sqrt(1e-5+sum(secants.^2)));

%setup
M = size(secants, 2);

P_k = zeros(N, N);
L_k = zeros(N, N);
funA_L_k = zeros(M,1); % because L_k is zero
q_k = zeros(M, 1);

r_rank = 0;

Lambda_k = zeros(N, N);
w_k = zeros(M, 1);

%outerloop
qiter = 0;
t1 = 0; t2 = 0; t3 = 0;



for ll=1:opt.outermost_iterations
    
    if (ll > 5)
        fprintf('Column generation step\n')
        %throw within-margin secants. keep at margin secants
        keepidx = find( abs((abs(q_k-1)-delta)) < delta/100 );
        fprintf('Keeping %d \n', length(keepidx));
        secants = secants(:, keepidx);
        w1 = w_k(keepidx); q1 = q_k(keepidx);
        idxStk = idxStk(keepidx, :);
        
        %search over a huge million secants to find violating secants
        sIter = 0;
        secNew = [];
        idxNew = [];
        tim1 = 0; 
        
        while (sIter < opt.num_cg_steps) && (size(secNew, 2) < opt.max_cg_secants)
            cg_tic = tic;
            sIter = sIter + 1;
            idx = randperm(size(data, 2));
            idx = idx(1:opt.num_cg_cols);
            
            
            Cmat = data(:, idx)'*P_k*data(:, idx);
            Cmat = diag(Cmat)*ones(1, size(Cmat, 2)) + ones(size(Cmat, 1),1)*diag(Cmat)' - 2*Cmat;
            
            Dmat = data(:, idx)'*data(:, idx);
            Dmat = diag(Dmat)*ones(1, size(Dmat, 2)) + ones(size(Dmat, 1),1)*diag(Dmat)' - 2*Dmat;
            
            [Xx, Yy] = meshgrid(1:length(idx), 1:length(idx));
            upp_lep = find(Xx > Yy); upp_lep = upp_lep(:);
            qval  = Cmat(upp_lep)./(1e-8+Dmat(upp_lep));
            idx1 = idx(Yy(upp_lep));
            idx2 = idx(Xx(upp_lep));
            
            keep_idx = find( abs(qval-1) > delta);
            if (length(keep_idx) > opt.max_cg_secants)
               [junk, k_idx] = sort(abs(qval-1), 'descend');
               keep_idx = k_idx(1:opt.max_cg_secants);
            end
            
            
            idx1 = idx1(keep_idx); idx2 = idx2(keep_idx);
            
            sec = data(:, idx1)-data(:, idx2);
            sec = sec*sparse(1:size(sec, 2), 1:size(sec, 2), 1./sqrt(1e-10+sum(sec.^2)), size(sec, 2), size(sec, 2));
            
            idxNew = [idxNew; idx1(:) idx2(:)];
            secNew = [secNew full(sec)];
            tim1 = tim1 + toc(cg_tic);
        end

        idxNew = sort(idxNew, 2);
        [idxNew, uidx] = unique(idxNew, 'rows');
        secNew = secNew(:, uidx);
        
        [C, I] = setdiff(idxNew, idxStk, 'rows');
        idxStk = [idxStk; C];
        secNew = secNew(:, I);
        
        lala = funA_secants_WY(P_k, secNew);
        secMinMax = [secMinMax; 1-min(lala) max(lala)-1];
        
        fprintf('Adding %d secants. Total time: %4d. Min %1.2f Max %1.2f \n', size(secNew, 2), round(tim1), min(lala), max(lala));
        
        secants = [secants secNew];
        
        w_k = [ w1; zeros(size(secNew, 2), 1) ];
        q_k = [ q1; zeros(size(secNew, 2), 1) ];
        
        M = length(q_k);
        
    end
    
    funA = @(z) funA_secants_WY(z, secants);
    funAT = @(z) funAT_secants_WY(z, secants);
    b = ones(M, 1);
    
    
    funA_L_k = funA(L_k);
    
    
    for qq=1:opt.outer_iterations
        qiter = qiter + 1;
        for kk=1:opt.inner_iterations
            
            %sing value thres.
            tic;
            [P_k1, r_rank] = solve_nuclear_norm_problem_approximate(L_k - Lambda_k, r_rank, opt.beta2);
            t3 = t3 + toc;
            
            %inf-norm solver
            tic
            q_k1 = solve_infinity_norm_problem(b, funA_L_k+w_k, delta);
            t1 = t1 + toc;
            
            %lin solver
            tic;
            switch opt.linear_solver
                case 'cgs'
                    L_k1 = solve_linear_problem( funA, funAT, q_k1-w_k, P_k1+Lambda_k, N, N, opt.beta1, 2*opt.beta2, opt.linear_iterations, L_k(:));
                case 'l-bfgs'
                    L_k1 = solve_linear_problem_lbfgs( funA, funAT, q_k1-w_k, P_k1+Lambda_k, N, N, opt.beta1, opt.beta2, opt.linear_iterations, opt.lbfgs_rank, L_k(:));
                case 'grad'
                    L_k1 = grad_descent(L_k, funA_L_k, funAT, q_k1-w_k, P_k1+Lambda_k, N, N, opt.beta1, 2*opt.beta2, opt.tau);
            end
            t2 = t2+toc;
            
            q_k = q_k1;
            L_k = L_k1; funA_L_k = funA(L_k1);
            P_k = P_k1;
            
        end
        
        
        Lambda_k1 = Lambda_k - opt.eta1*(L_k1 - P_k1);
        w_k1  = w_k - opt.eta2*(q_k1 - funA_L_k);
        
        w_k = w_k1;
        Lambda_k = Lambda_k1;
        
        
        if (opt.display) %mod(qiter, 5) == 0 % turn on/off plotting
            subplot 211
            plot(b, 'k-.'); hold on
            plot(q_k1, 'r*');
            plot(funA(L_k1), 'go');
            hold off
            axis tight
            
            
            subplot 212
            if (ll > 5)
                plot(secMinMax);
            end
            
            drawnow
        end
        
        
        if mod(qiter,5) == 0 % report progress and check for stopping every 5 iterations
            err_q  = 2*norm(q_k1 - funA_L_k)/(norm(funA_L_k)+norm(q_k1));
            err_L = 2*norm(L_k1-P_k1, 'fro')/(norm(L_k1, 'fro')+norm(P_k1, 'fro'));
            
            if (opt.verbose)
                fprintf('itr:%03d', qiter);
                fprintf(' rnk:%03d err:%4.2e', rank(P_k), norm(q_k1-b,inf));
                fprintf(' Q-L:%4.2e L-P:%4.2e', err_q, err_L);
                fprintf(' sec: %4.2f %4.2f %4.2f\n', t1/qiter,t2/qiter,t3/qiter);
            end
            
            if (ll > 5) && ( max(err_L,err_q) < opt.tol)
                return
            end
            
        end
    end
    
end

end

function L_k1 = grad_descent(L_k, funA_L_k, funAT, b, Blah1, N1, N2, lambda, beta, tau)

g =   lambda*funAT(funA_L_k-b) ...
    + beta*(L_k - Blah1); % gradient

L_k1 = L_k - tau*g; % gradient descent

end


function L_k1 = solve_linear_problem( funA, funAT, b, Blah1, N1, N2, lambda, beta, iter, x0);

tval = lambda*funAT(b)+beta*Blah1;
mycolon = @(z) z(:);

Afun = @(z) (mycolon( lambda*funAT(funA(reshape(z, [N1 N2])))) + beta*z);
[L_k1, flag] = cgs(Afun, tval(:), 1e-5, iter, [], [], x0);

L_k1 = reshape(L_k1, [N1 N2]);

end

function L_k1 = solve_linear_problem_lbfgs( funA, funAT, b, Blah1, N1, N2, beta1, beta2, iter, rrank, x0);

funObj = @(xx, vv) function_handle_lbfgs(xx, funA, funAT, b, Blah1, beta1, beta2);
opt.Display = 'iter';
opt.MaxIter = iter;

opt.Display = 'off'; %'final';
opt.optTol = 1e-5;
opt.progTol = 1e-6;
opt.Corr = rrank;

L_k1 = minFunc(funObj, x0(:), opt, []);
L_k1 = reshape(L_k1, [N1 N2]);

end

function [fval, gval] = function_handle_lbfgs(x, funA, funAT, b, Q, beta1, beta2);
x = reshape(x, size(Q));
tmp1 = funA(x) - b;
fval = beta1*norm(tmp1)^2 + beta2*norm(x - Q, 'fro')^2;

gval = 2*beta1*funAT(tmp1)+2*beta2*(x-Q);
gval = gval(:);

end

function P_k1 = solve_nuclear_norm_problem(A, beta)

[U, S, V] = svd(A);

S = diag(S);
S = S - 1/beta;
S(S < 0) = 0;
P_k1 = U*diag(S)*V';

end


function [P_k1, r_rank] = solve_nuclear_norm_problem_approximate(A, r_rank, beta)

[m,n] = size(A);
if max(m,n) < 2000
    
    [U, S, V] = svds(A, r_rank+5);
    S = diag(S);
else
    sn = min([m/2,n/2,round(2*(r_rank+5)-2)]);
    [U,S] = LinearTimeSVD(A, sn, r_rank+5, ones(n,1)/n);
    invS = zeros(size(S)); indx = (S>0);
    invS(indx) = 1./S(indx);
    V = (spdiags(invS,0,numel(invS),numel(invS))*U.'*A).';
end


S = S - 1/beta;
S(S < 0) = 0;
P_k1 = U*spdiags(S,0,size(U,2),size(V,2))*V';
r_rank = nnz(S>0);

end

function x = solve_infinity_norm_problem(b, y, delta)

y1 = y - b;

ysign = sign(y1);

x = ysign.*min(delta, abs(y1));
x = x + b;

end

function opt = getDefaultParameters(opt)
if ~isfield(opt, 'verbose')
    opt.verbose = 1;
end
if ~isfield(opt, 'display')
    opt.display = 0;
end
if ~isfield(opt, 'tol')
    opt.tol = 1e-4;
end
if ~isfield(opt, 'inner_iterations')
    opt.inner_iterations = 1;
end
if ~isfield(opt, 'outer_iterations')
    opt.outer_iterations = 10;
end
if ~isfield(opt, 'outermost_iterations')
    opt.outermost_iterations = 1000;
end
if ~isfield(opt, 'beta1')
    opt.beta1 = 1;
end
if ~isfield(opt, 'beta2')
    opt.beta2 = 1;
end
if ~isfield(opt, 'eta1')
    opt.eta1 = 1.6;
end
if ~isfield(opt, 'eta2')
    opt.eta2 = 1.6;
end
if ~isfield(opt, 'linear_solver')
    opt.linear_solver = 'cgs';
end
switch opt.linear_solver
    case 'cgs'
        if ~isfield(opt, 'linear_iterations')
            opt.linear_iterations = 10;
        end
    case 'l-bfgs'
        if ~isfield(opt, 'linear_iterations')
            opt.linear_iterations = 10;
        end
        if ~isfield(opt, 'lbfgs_rank')
            opt.lbfgs_rank = 10;
        end
    case 'grad'
        if ~isfield(opt, 'tau')
            opt.tau = 1e-1;
        end
end
end
