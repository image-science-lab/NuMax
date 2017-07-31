function P = funAT_secants_WY(b, secants)
M = size(secants, 2);
P = secants*spdiags(b,0,M,M)*secants.';
