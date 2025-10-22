function out = fromSqrtInfo(varargin)

if nargin == 1
    Xi = varargin{1};
    nu = zeros(size(Xi, 2), 1);
elseif nargin == 2
    nu = varargin{1};
    Xi = varargin{2};
else
    error('Expected 1 or 2 arguments');
end
assert(istriu(Xi), 'Expected Xi to be upper triangular');
assert(iscolumn(nu), 'Expected nu to be a column vector')
assert(length(nu) == size(Xi, 2), 'Expected dimensions of nu and Xi to be compatible');

s_ut = struct('UT', true, 'TRANSA', false);         % for solving S*x = b for x with upper triangular S
s_ut_transa = struct('UT', true, 'TRANSA', true);   % for solving S.'*x = b for x with upper triangular S

% Solve Xi*mu = nu
mu = linsolve(Xi, nu, s_ut); % Xi\nu (triangular backward substitution)

% S = qr(Xi^{-T})
L = linsolve(Xi, eye(size(Xi, 2)), s_ut_transa); % Xi.'\I (triangular inverse via forward substitution)
S = qr(L, "econ");
out = Gaussian(mu, S);
