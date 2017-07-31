function [P_k1, r_rank, L_k1, q_k1, Lambda_k1, w_k1] = NuMax(funA, funAT, b, delta, opt)
%%%%%%
% solves the following problem
% P_k1 = arg min trace(P)   s.t  \| A(P) - b \|_\infty < \delta, P >= 0
%
% Input
%   funA: A operator. see problem definition above
%   funAT: Adjoint of A
%   b, delta: see problem definition above
% Output
%   P_k1: Solution to the optmization problem
%   r_rank: Rank of the solution
%
% Parameters
%  opt.verbose - verbosity: 0 or 1*
%  opt.display - graphic plots: 0* or 1
%  opt.inner_iterations: # of inner iterations. Default: 1
%  opt.outer_iterations: # of outer iterations. Default: 1000
%  opt.beta1, opt.beta2: Penalty parameters (see paper). Default: 1
%  opt.eta1, opt.eta2: Lagrange update paramaters (see paper). Default: 1.6
%  opt.linear_solver -- least squares solver
%        'cgs' - conjugate gradients. also define opt.linear_iterations (Default)
%        'l-bfgs' - Limited memory BFGS. also define opt.linear_iterations and opt.lbfgs_rank
%        'grad'  -  gradient descent. also define opt.tau
if ~exist('opt')
    opt = [];
end
opt = getDefaultParameters(opt);

junk = funAT(b);

[N1, N2] = size(junk);
M = length(b);

%setup
P_k = zeros(N1, N2);
L_k = zeros(N1, N2);
funA_L_k = zeros(M,1); % because L_k is zero
q_k = zeros(M, 1);

r_rank = 0;

Lambda_k = zeros(N1, N2);
w_k = zeros(M, 1);

%outerloop
t1 = 0; t2 = 0; t3 = 0;
for qq=1:opt.outer_iterations
    
    for kk=1:opt.inner_iterations
        %innerloop
        %sing value thres.
        tic;
        [P_k1, r_rank] = solve_nuclear_norm_problem_approximate(L_k - Lambda_k, r_rank, opt.beta2);
%        [P_k1, r_rank] = solve_EVP_problem_approximate(L_k - Lambda_k, r_rank, opt.beta2);
        
        t3 = t3 + toc;
        
        %inf-norm solver
        tic
        q_k1 = solve_infinity_norm_problem(b, funA_L_k+w_k, delta);
        t1 = t1 + toc;
        
        %lin solver
        tic;
        switch opt.linear_solver
            case 'cgs'
                L_k1 = solve_linear_problem( funA, funAT, q_k1-w_k, P_k1+Lambda_k, N1, N2, opt.beta1, 2*opt.beta2, opt.linear_iterations, L_k(:));
            case 'l-bfgs'
                L_k1 = solve_linear_problem_lbfgs( funA, funAT, q_k1-w_k, P_k1+Lambda_k, N1, N2, opt.beta1, opt.beta2, opt.linear_iterations, opt.lbfgs_rank, L_k(:));
            case 'grad'
                L_k1 = grad_descent(L_k, funA_L_k, funAT, q_k1-w_k, P_k1+Lambda_k, N1, N2, opt.beta1, 2*opt.beta2, opt.tau);
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
    
    if opt.display % turn on/off plotting
        plot(b, 'k-.'); hold on
        plot(q_k1, 'r*');
        plot(funA(L_k1), 'go');
        hold off
        axis tight
        drawnow
    end
    
    if  (mod(qq,5) == 0) % report progress and check for stopping every 5 iterations
        err_q  = 2*norm(q_k1 - funA_L_k)/(norm(funA_L_k)+norm(q_k1));
        err_L = 2*norm(L_k1-P_k1, 'fro')/(norm(L_k1, 'fro')+norm(P_k1, 'fro'));
        
        if (opt.verbose)
            fprintf('itr:%03d', qq);
            fprintf(' rnk:%03d err:%4.2e', rank(P_k), norm(q_k1-b,inf));
            fprintf(' Q-L:%4.2e L-P:%4.2e', err_q, err_L);
            fprintf(' sec: %4.2f %4.2f %4.2f\n', t1/qq,t2/qq,t3/qq);
        end
        
        if (max(err_L,err_q) < opt.tol)
            return
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


function [P_k1, r_rank] = solve_EVP_problem_approximate(A, r_rank, beta)

[m,n] = size(A);
[U, S] = eig((A+A')/2);
S  = diag(S);
S = S - 1/beta;
S(S < 0) = 0;
P_k1 = U*spdiags(S,0,size(U,2),size(U,2))*U';
r_rank = nnz(S>0);

end

function [P_k1, r_rank] = solve_nuclear_norm_problem_approximate(A, r_rank, beta)

[m,n] = size(A);
if max(m,n) < 1000
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
