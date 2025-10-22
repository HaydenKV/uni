function out = fromMoment(varargin)

if nargin == 1
    P = varargin{1};
    mu = zeros(size(P, 2), 1);
elseif nargin == 2
    mu = varargin{1};
    P = varargin{2};
else
    error('Expected 1 or 2 arguments');
end
assert(size(P, 1) == size(P, 2), 'Expected P to be square');
assert(issymmetric(P), 'Expected P to be symmetric');
assert(iscolumn(mu), 'Expected mu to be a column vector')
assert(length(mu) == size(P, 2), 'Expected dimensions of mu and P to be compatible');

% Let S be an upper-triangular matrix such that S^T*S = Lambda
S = chol(P);

out = GaussianInfo.fromSqrtMoment(mu, S);
