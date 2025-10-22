function out = conditional(obj, idxA, idxB, dataB)

% if dataB = xB is a column vector:
% Given joint density p(x), return conditional density p(x(idxA) | x(idxB) = xB)

% if dataB = p(x(idxB) | y) is a GaussianInfo object:
% Given joint density p(x), return conditional density p(x(idxA) | y)

nA = length(idxA);
nB = length(idxB);
n = nA + nB;
assert(n == obj.dim());

RR = qr([obj.Xi(:, idxA), obj.Xi(:, idxB), obj.nu], "econ");
% Q-less QR of [Xi(:, idxA), Xi(:, idxB), nu] yields
% [R1, R2, nu1;
%   0, R3, nu2]

R1 = RR(1:nA, 1:nA);
R2 = RR(1:nA, nA+1:n);
% R3 = RR(nA+1:n, nA+1:n);

nu1 = RR(1:nA, n+1);
% nu2 = RR(nA+1:n, n+1);

switch class(dataB)
    case 'GaussianInfo'
        SS = qr([R2, R1, nu1; dataB.Xi, zeros(nB, nA), dataB.nu], "econ");
        % Q-less QR yields
        % [S1, S2, s1;
        %   0, S3, s2]
        % p(x(idxA) | y) = N^-0.5(x(idxA); s2, S3)
        nuc = SS(nB+1:n, n+1);
        Xic = SS(nB+1:n, nB+1:n);
    case 'double'
        % p(x(idxA) | x(idxB) = xB) = N^-0.5(x(idxA); nu1 - R2*xB, R1)
        nuc = nu1 - R2*dataB;
        Xic = R1;
    otherwise
        error('Expected data to be a column vector or a GaussianInfo object');
end

out = GaussianInfo(nuc, Xic);
