% This helper function evaluates the map
% [         x[k] ] |---> [ x[k]   ]
% [  dw(idxQ)[k] ] |     [ x[k+1] ]
% for one step of RK4 integration
function [xa, Ja] = RK4SDEHelperAugmented(obj, xdw, dt, idxQ)

nx = obj.density.dim();             % Dimension of state
nq = length(idxQ);                  % Dimension of process noise
x = xdw(1:nx);

switch nargout
    case 1
        xnext = RK4SDEHelper(obj, xdw, dt, idxQ);
    case 2
        [xnext, J] = RK4SDEHelper(obj, xdw, dt, idxQ);
        Jx = [eye(nx), zeros(nx, nq)];
        % Ja = [Jx(idxQ, :); J];
        Ja = [Jx; J(idxQ, :)];
        % Ja = [Jx; J];
end
% xa = [x(idxQ); xnext];
xa = [x; xnext(idxQ)];
% xa = [x; xnext];
