function out = unscentedTransform(obj, h)

n = obj.dim();              % Length of input
nsigma = 2*n + 1;           % Number of sigma points

% Unscented transform parameters
alpha = 1;
kappa = 0;
lambda = alpha^2*(n + kappa) - n;
gamma = realsqrt(n + lambda);
beta = 2;

% Generate sigma points
mu = obj.mean();
gammaST = gamma*obj.sqrtCov().';
X = [mu + gammaST, mu - gammaST, mu];

% Transform the sigma points through the function
muy = h(X(:, nsigma));
ny = length(muy);
Y = zeros(ny, nsigma);
Y(:, nsigma) = muy;
for i = 1:nsigma-1
    Y(:, i) = h(X(:, i));
end

% Unscented mean
wm = [repmat(1/(2*(n + lambda)), 1, 2*n), lambda/(n + lambda)];
muy = sum(wm.*Y, 2);

% Compute unscented mean and sqrt covariance
wc = [repmat(1/(2*(n + lambda)), 1, 2*n), lambda/(n + lambda) + (1 - alpha^2 + beta)];
dY = realsqrt(wc).*(Y - muy);
Syy = qr(dY.', "econ");
out = GaussianInfo.fromSqrtMoment(muy, Syy);
