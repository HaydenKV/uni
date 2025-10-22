function out = fromInfo(varargin)

if nargin == 1
    Lambda = varargin{1};
    eta = zeros(size(Lambda, 2), 1);
elseif nargin == 2
    eta = varargin{1};
    Lambda = varargin{2};
else
    error('Expected 1 or 2 arguments');
end
assert(size(Lambda, 1) == size(Lambda, 2), 'Expected Lambda to be square');
assert(issymmetric(Lambda), 'Expected Lambda to be symmetric');
assert(iscolumn(eta), 'Expected eta to be a column vector')
assert(length(eta) == size(Lambda, 2), 'Expected dimensions of eta and Lambda to be compatible');

s_ut_transa = struct('UT', true, 'TRANSA', true);   % for solving S.'*x = b for x with upper triangular S

% Let Xi be an upper-triangular matrix such that Xi^T*Xi = Lambda
Xi = chol(Lambda);

% Solve Xi^T*nu = eta
nu = linsolve(Xi, eta, s_ut_transa);
out = GaussianInfo.fromSqrtInfo(nu, Xi);
