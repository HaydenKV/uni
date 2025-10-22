function Xi = sqrtInfoMat(obj)

% Xi = qr(S^{-T})
L = linsolve(obj.S, eye(obj.dim()), obj.s_ut_transa); % S.'\I (triangular inverse via forward substitution)
Xi = qr(L, "econ");
