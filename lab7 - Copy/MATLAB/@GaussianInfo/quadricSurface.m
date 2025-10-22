function Q = quadricSurface(obj, n_sigma)

arguments
    obj (1, 1) GaussianInfo
    n_sigma (1, 1) double = 3
end

n = obj.dim();
assert(n == 3, 'Expected trivariate Gaussian');

c = 2*normcdf(n_sigma) - 1;                     % Probability mass enclosed by n_sigma standard deviations (same as normcdf(n_sigma) - normcdf(-n_sigma))
r2 = chi2inv(c, n);                             % Squared radius in w coords

y = -obj.infoVec();
Q = [obj.infoMat(), y; y.', obj.nu.'*obj.nu - r2];

