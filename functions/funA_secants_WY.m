function b = funA_secants_WY(P, secants)
N = size(secants, 1);
P = reshape(P, N, N);
b = sum(secants.*(reshape(P, N, N)*secants),1).';
