function [H,sigma] = LinearTimeSVD(A,c,k,p)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% Reference: P. Drineas, R. Kannan, M. Mahoney, 
% "Fast Monte Carlo Alrorithms for Matrices II: 
% Computing a Low-Rank Approximation to a Matrix"
%
% Author: Shiqian Ma. 
% Date: June. 02, 2008.
% Slightly modified by Wotao Yin in 2012 for higher efficiency
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get dimension
[m,n] = size(A);
c = round(c); k = round(k);
% checking inputs
Check = (k>=1)&&(k<=c)&&(c<=n)&&(min(p)>=0);%&(sum(p)==1);
if ~Check 
    fprintf('k = %d, c = %d, n = %d, min(p) = %f\n', k, c, n, min(p));
    error('inputs error!\n');
end
%% main iteration
C = zeros(m,c);
bp = cumsum(p);
for t = 1: c
    ind = find(bp >= rand, 1, 'first');
    C(:,t) = A(:,ind)/sqrt(c*p(ind));
end
[U,S,junk] = svd(C'*C);
sigma = sqrt(diag(S));
sigma = sigma(1:k);
% sigma = sigma(sigma>1e-4);
k = length(sigma);
H = zeros(m,k);
for t = 1: k
    h = C*U(:,t);
    nh = norm(h);
    if nh == 0
        H(:,t) = zeros(size(h));
    else
        H(:,t) = h/nh;
    end
end