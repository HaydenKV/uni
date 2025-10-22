function X = simulate(obj, m)

arguments (Input)
    obj (1, 1) GaussianBase
    m (1, 1) uint64 = 1             % Number of samples to generate
end

arguments (Output)
    X (:, :) double                 % n-by-m matrix of samples, where n = obj.dim()
end

% Draw m realisations of a Gaussian random variable
n = obj.dim();
X = nan(n, m);
% TODO: Merge from MCHA4100
