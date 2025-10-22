function Lambda = infoMat(obj)

% Lambda = (S^T*S)^{-1} = S^{-1}*S^{-T}
L = linsolve(obj.S, eye(obj.dim()), obj.s_ut_transa); % S.'\I (triangular inverse via forward substitution)
Lambda = L.'*L;
