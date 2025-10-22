function Q = quadricSurface(obj, n_sigma)

arguments (Input)
    obj (1, 1) Gaussian
    n_sigma (1, 1) double = 3
end

arguments (Output)
    Q (4, 4) double
end

n = obj.dim();
assert(n == 3, 'Expected trivariate Gaussian');

c = 2*normcdf(n_sigma) - 1;                     % Probability mass enclosed by n_sigma standard deviations (same as normcdf(n_sigma) - normcdf(-n_sigma))
r2 = chi2inv(c, n);                             % Squared radius in w coords

Sti = linsolve(obj.S, eye(n), obj.s_ut_transa); % inv(S.')
z = linsolve(obj.S, obj.mu, obj.s_ut_transa);   % S.'\mu (triangular forward substitution)
y = -linsolve(obj.S, z, obj.s_ut);              % -S\z (triangular backward substitution)
Q = [Sti.'*Sti, y; y.', z.'*z - r2];

