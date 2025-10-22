function out = marginal(obj, idx)

idxNot = setdiff(1:obj.dim(), idx); % Complementary indices
nI = length(idx);
nNotI = length(idxNot);
n = nI + nNotI;
assert(n == obj.dim());

RR = qr([obj.Xi(:, idxNot), obj.Xi(:, idx), obj.nu], "econ");
% Q-less QR of [Xi(:, idxNot), Xi(:, idx), nu] yields
% [R1, R2, nu1;
%   0, R3, nu2]

R3 = RR(nNotI+1:n, nNotI+1:n);
nu2 = RR(nNotI+1:n, n+1);

% p(x(idx)) = N^-0.5(x(idx); nu2, R3)
num = nu2;
Xim = R3;
out = GaussianInfo(num, Xim);
