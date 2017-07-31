function b = funA_secants_circ(P, Amat, secants_fourier)

N = size(secants_fourier, 1);
P = reshape(P, N, N);
b = sum(conj(secants_fourier).*((Amat.*P)*secants_fourier), 1).';

b = real(b);