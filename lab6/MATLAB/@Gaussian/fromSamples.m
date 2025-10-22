function out = fromSamples(X)

arguments (Input)
    X (:, :) double                 % n-by-m matrix of samples, where n is the dimension and m is the number of samples
end

arguments (Output)
    out (1, 1) Gaussian
end

[n, m] = size(X);

% Compute the sample mean
mu = nan(n, 1);

% Compute the sample square-root covariance
S = nan(n, n);

out = Gaussian(mu, S);
% TODO: Merge from MCHA4100 or implement a square-root version directly using a QR decomposition

