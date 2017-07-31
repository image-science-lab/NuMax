function P = funAT_secants_circ(b,  Amat, secants_fourier)
M = size(secants_fourier, 2);


P = secants_fourier*spdiags(b,0,M,M)*secants_fourier';
P = conj(Amat).*P;