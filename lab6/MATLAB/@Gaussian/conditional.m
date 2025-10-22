function out = conditional(obj, idxA, idxB, dataB)

% if dataB = xB is a column vector:
% Given joint density p(x), return conditional density p(x(idxA) | x(idxB) = xB)

% if dataB = p(x(idxB) | y) is a Gaussian object:
% Given joint density p(x), return conditional density p(x(idxA) | y)

SS = qr([obj.S(:, idxB), obj.S(:, idxA)], "econ");

nA = length(idxA);
nB = length(idxB);

SBB = SS(1:nB, 1:nB);
SBA = SS(1:nB, nB+1:end);
SAA = SS(nB+1:end, nB+1:end);

muA = obj.mu(idxA);
muB = obj.mu(idxB);

switch class(dataB)
    case 'Gaussian'
        K = linsolve(SBB, SBA, obj.s_ut); % K = SBB\SBA; % triangular backward substitution
        Sc = qr([dataB.S*K; SAA], "econ");
        muc = muA + K.'*(dataB.mu - muB);
    case 'double'
        Sc = SAA;
        muc = muA + SBA.'*linsolve(SBB, dataB - muB, obj.s_ut_transa);  % muc = muA + SBA.'*(SBB.'\(xB - muB)); % triangular forward substitution
    otherwise
        error('Expected data to be a column vector or a Gaussian object');
end

out = Gaussian(muc, Sc);
