function eta = infoVec(obj)

% eta = (S^T*S)^{-1}*mu = S^{-1}*S^{-T}*mu
z = linsolve(obj.S, obj.mu, obj.s_ut_transa); % S.'\mu (triangular forward substitution)
eta = linsolve(obj.S, z, obj.s_ut); % S\z (triangular backward substitution)
