function S = sqrtCov(obj)

% S = qr(Xi^{-T})
L = linsolve(obj.Xi, eye(size(obj.Xi, 2)), obj.s_ut_transa); % Xi.'\I (triangular inverse via forward substitution)
S = qr(L, "econ");
