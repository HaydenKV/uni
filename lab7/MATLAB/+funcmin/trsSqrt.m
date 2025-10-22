% Trust-region subproblem (square-root Hessian)
% minimise 0.5*p.'*H*p + g.'*p subject to ||z|| < Delta
% where z = Xi*p and
%       Xi is an upper triangular matrix such that Xi.'*Xi = H
function [p, ret] = trsSqrt(Xi, g, Delta)

arguments
    Xi (:, :) double
    g (:, 1) double
    Delta (1, 1) double
end

assert(size(Xi, 1) == size(Xi, 2));
assert(size(Xi, 1) == size(g, 1));
assert(istriu(Xi));

% Cache structure used by linsolve
persistent s_ut s_ut_transa
if isempty (s_ut) || isempty(s_ut_transa)
    s_ut = struct('UT', true, 'TRANSA', false);         % for solving S*x = b for x with upper triangular S
    s_ut_transa = struct('UT', true, 'TRANSA', true);   % for solving S.'*x = b for x with upper triangular S
end

% Solve Xi.'*gtilde = g for gtilde
gtilde = linsolve(Xi, g, s_ut_transa); % Xi.'\g;

% Step length
alpha = min(1.0, Delta/norm(gtilde));

% Solve Xi*p = -alpha*gtilde for p
p = -alpha*linsolve(Xi, gtilde, s_ut); % -alpha*(Xi\gtilde);

ret = 0;
