function [P_k, r_rank, R_k, Q_k] = NuMax_Dict_v1(Dict, mu, alpha, opt)

[M, N] = size(Dict);

DDT = Dict*Dict';
P_k = zeros(M, M); 
R_k = zeros(M, M); DRD = zeros(N, N); S_k = zeros(M, M);
Q_k = zeros(N, N); L_k = zeros(N, N);

r_rank = 0;


for iter = 1:opt.outer_iterations
    
    for ee = 1:opt.inner_iterations
        %Q-update
        Q_k1 = DRD + L_k;
        qdiags = diag(Q_k1);
        Q_k1 = sign(Q_k1).*min(abs(Q_k1), mu);
        Q_k1 = Q_k1 - diag(diag(Q_k1));
        Q_k1 = Q_k1 + diag(alpha*ones(N, 1));
        
        %R-update
        mycolon = @(x) x(:);
        funA = @(x) opt.beta1*x(:) + opt.beta2*mycolon(DDT*reshape(x, size(R_k))*DDT);
        rhs_term = opt.beta1*(P_k - S_k) + opt.beta2*Dict*(Q_k1-L_k)*Dict';
        [R_k1, flag] = cgs(funA, rhs_term(:), 1e-6, 100);
        R_k1 = reshape(R_k1, size(R_k));
        
        DRD = Dict'*R_k1*Dict;
        
        %P-update
        %[U_tmp, S_tmp, V_tmp] = svds( ((R_k1+S_k)+(R_k1+S_k)')/2, r_rank+5);
        %S_tmp = diag(S_tmp);
        %S_tmp = max(0, S_tmp-1/opt.beta1);
        %P_k1 = U_tmp*spdiags(S_tmp,0,size(U_tmp,2),size(V_tmp,2))*V_tmp';
        [Vtmp, S_tmp] = eigs(((R_k1+S_k)+(R_k1+S_k)')/2, r_rank+5, 1e6);
        S_tmp = diag(S_tmp);
        S_tmp = max(0, S_tmp-1/opt.beta1);
        P_k1 = Vtmp*spdiags(S_tmp,0,size(Vtmp,2),size(Vtmp,2))*Vtmp';
        r_rank = nnz(S_tmp>0);

        P_k = P_k1; Q_k = Q_k1; R_k = R_k1;
    end
    
    S_k = S_k - opt.eta*(P_k - R_k);
    L_k = L_k - opt.eta*(Q_k - DRD);
    
    e1 = 2*norm(P_k - R_k, 'fro')/(1e-8+norm(P_k, 'fro')+norm(R_k, 'fro'));
    e2 = 2*norm(Q_k - DRD, 'fro')/(1e-8+norm(Q_k, 'fro')+norm(DRD,'fro'));
    fprintf('Iter: %03d ', iter);
    fprintf('Rank: %d ', r_rank);
    fprintf('P-R: %1.3f Q-R: %1.3f ', e1, e2);
    fprintf('\n');
    if (e1+e2 < 1e-3)
        break;
    end
    
end