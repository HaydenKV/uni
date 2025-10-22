function X = confidenceEllipse(obj, n_sigma, n_samples)

arguments (Input)
    obj (1, 1) GaussianBase
    n_sigma (1, 1) double = 3       % number of standard deviations
    n_samples (1, 1) uint64 = 100   % number of samples
end

arguments (Output)
    X (2, :) double                 % 2-by-n_samples matrix of points on boundary of confidence ellipse
end

n = obj.dim();
assert(n == 2, 'Expected bivariate Gaussian');

c = nan;                            % Probability mass enclosed by n_sigma standard deviations
r = nan;                            % Radius in w coords
t = linspace(0, 2*pi, n_samples);   % Sampling angles for circle
W = r*[cos(t); sin(t)];             % Circle sampling points in w coords
X = nan(size(W));                   % Points on ellipse in x coords
% TODO: Merge from MCHA4100
