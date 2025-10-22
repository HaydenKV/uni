% Trust-region subproblem (square-root inverse Hessian)
% minimise 0.5*p.'*H*p + g.'*p subject to ||z|| < Delta
% where z is the solution to S.'*z = p and
%       S is an upper triangular matrix such that S.'*S = inv(H)
function [p, ret] = trsSqrtInv(S, g, Delta)

arguments
    S (:, :) double
    g (:, 1) double
    Delta (1, 1) double
end

assert(size(S, 1) == size(S, 2));
assert(size(S, 1) == size(g, 1));

gtilde = S*g;

% Step length
alpha = min(1.0, Delta/norm(gtilde));

p = -alpha*S.'*gtilde;

ret = 0;
