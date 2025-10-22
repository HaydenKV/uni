% Trust-region subproblem (square-root Hessian)
% minimise 0.5*p.'*H*p + g.'*p subject to ||z|| < Delta
% where z = Xi*Pi.'*p and
%       Xi is an upper triangular matrix such that Pi*Xi.'*Xi*Pi.' = H
function [p, ret] = trsSqrtSparse(Xi, pidx, g, Delta)

arguments
    Xi (:, :) double {mustBeSparse}
    pidx (1, :) double
    g (:, 1) double
    Delta (1, 1) double
end

assert(size(Xi, 1) == size(Xi, 2));
assert(size(Xi, 1) == size(g, 1));
assert(length(pidx) == size(Xi, 2));
assert(istriu(Xi));

% Solve Xi.'*gtilde = Pi.'*g = I(:, pidx).'*g = I(pidx, :)*g = g(pidx) for gtilde
gtilde = Xi.'\g(pidx); % Sparse triangular solve

% Step length
alpha = min(1.0, Delta/norm(gtilde));

% Solve Xi*Pi.'*p = Xi*I(:, pidx).'*p = Xi*I(pidx, :)*p = Xi*p(pidx) = -alpha*gtilde for p
p = zeros(size(Xi, 2), 1);
p(pidx) = -alpha*(Xi\gtilde); % Sparse triangular solve

ret = 0;
