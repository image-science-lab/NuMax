function [delta_min, delta_max] = get_rip_constants(data, phi)
%%%
% Given 
%  - data: data matrix of size N x d, where N is the dimension of the
%          datapoint and p is the # of datapoints.
%  - phi: measurement matrix of size M x N
%
% Code generates all pairwise secants of data and 
%  reports lower and upper isometry constants

N = size(phi, 2);
M = size(phi, 1);
T = size(data, 2);

Dmat = data'*data;
Dmat = diag(Dmat)*ones(1, T)+ones(T,1)*diag(Dmat)' -2*Dmat;

Cmat = (phi*data)'*(phi*data); 
Cmat = diag(Cmat)*ones(1,T)+ones(T,1)*diag(Cmat)'-2*Cmat;

Cmat = Cmat+eye(T);
Dmat = Dmat+eye(size(Dmat));
Cmat = Cmat./(1e-8+Dmat);

delta_min = 1-min(Cmat(:));
delta_max = max(Cmat(:))-1;