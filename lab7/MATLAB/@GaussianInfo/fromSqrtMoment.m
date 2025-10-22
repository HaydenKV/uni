function out = fromSqrtMoment(varargin)

if nargin == 1
    S = varargin{1};
    mu = zeros(size(S, 2), 1);
elseif nargin == 2
    mu = varargin{1};
    S = varargin{2};
else
    error('Expected 1 or 2 arguments');
end
assert(istriu(S), 'Expected S to be upper triangular');
assert(iscolumn(mu), 'Expected mu to be a column vector')
assert(length(mu) == size(S, 2), 'Expected dimensions of mu and S to be compatible');

s_ut_transa = struct('UT', true, 'TRANSA', true);   % for solving S.'*x = b for x with upper triangular S

% Xi = qr(S^{-T})
L = linsolve(S, eye(size(S, 2)), s_ut_transa); % S.'\I (triangular inverse via forward substitution)
Xi = qr(L, "econ");

% Xi*mu = nu
nu = Xi*mu;

out = GaussianInfo(nu, Xi);
