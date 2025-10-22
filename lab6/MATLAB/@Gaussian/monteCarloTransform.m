function [out, Y, X] = monteCarloTransform(obj, h, n_samples)

nx = obj.dim();

if nargin < 3
    n_samples = 100^nx;         % 100 samples per input dimension
end

X = obj.simulate(n_samples);    % Draw n_samples realisations

% Propagate first sample through function to determine output dimension
y_first = h(X(:, 1));
ny = length(y_first);

Y = nan(ny, n_samples);
Y(:, 1) = y_first;

% Propagate remaining samples through function
for k = 2:n_samples
    Y(:, k) = h(X(:, k));       
end
out = Gaussian.fromSamples(Y);