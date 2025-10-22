function out = fromSamples(X)

%
% Input arguments
%
% X:        samples (n x m)
%
% where
%   n is the dimension
%   m is the number of samples
%
% Output arguments
%
% out:      Gaussian object
%

[n, m] = size(X);

% Compute the sample mean
mu = sum(X, 2)/m; % mean(X, 2)

% Compute the sample square-root covariance
S = qr(realsqrt(1/(m - 1))*(X - mu).', "econ");
out = GaussianInfo.fromSqrtMoment(mu, S);
