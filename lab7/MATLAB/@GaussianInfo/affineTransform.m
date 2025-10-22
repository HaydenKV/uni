function out = affineTransform(obj, h)

mux = obj.mean();
[muy, J] = h(mux);      % Evaluate function at mean value
[m, n] = size(J);

% Linearise y = h(x) about x = mux
% y ~= h(mux) + J*(x - mux)
%    = J*x + h(mux) - J*mux
b = muy - J*mux;

% SVD approach
[U, s, V] = svd(J, "vector");
tol = max(m, n)*eps(max(s));
r = nnz(s > tol);               % Rank
s1 = s(1:r);
U1 = U(:, 1:r);
U2 = U(:, r+1:m);
V1 = V(:, 1:r);
V2 = V(:, r+1:n);
Jp = (V1./s1(:).')*U1.';
X = obj.Xi*V2;
Y = obj.Xi*Jp;

%{
% COD approach
[U, R, p] = qr(J, "vector");
if m == 1   % If R has one row, diag(R) creates a diagonal matrix instead of returning R(1, 1)
    s = abs(R(1, 1));
else
    s = abs(diag(R));
end
tol = max(m, n)*eps(max(s));
r = nnz(s > tol);               % Rank
TT = qr([R(1:r, :).', obj.Xi(:, p).']);
T = TT(:, 1:r).';               % T is lower triangular
XiV = TT(:, r+1:end).';         % Xi*V
T11 = T(1:r, 1:r); 
U1 = U(:, 1:r);
U2 = U(:, r+1:m);
X = XiV(:, r+1:n);              % Xi*V2
Y = XiV(:, 1:r)/T11*U1.';       % Xi*V1*inv(T11)*U1.' (triangular solve)
%}

sigma_max_ub = realsqrt(sum([X, Y].^2, 'all')); % Cheap upper bound for the largest singular value of [X, Y]
kappa = 1e7*sigma_max_ub;   % Constraint information factor (kappa >> sigma_max_ub)

RR = qr([X, Y, obj.nu + Y*b; ...
    zeros(m - r, n - r), kappa*U2.', kappa*U2.'*b]);
% Q-less QR yields
% [R1, R2, nu1;
%   0, R3, nu2]
R3 = RR(n-r+1:n-r+m, n-r+1:n-r+m);
nu2 = RR(n-r+1:n-r+m, n-r+m+1);

% p(y) = N^-0.5(y; nu2, R3)
out = GaussianInfo(nu2, R3);
