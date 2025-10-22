% This helper function evaluates the map
% [         x[k] ] |---> [ x[k+1] ]
% [  dw(idxQ)[k] ] |
% for one step of RK4 integration.
function [xnext, J] = RK4SDEHelper(obj, xdw, dt, idxQ)
nx = obj.density.dim();             % Dimension of state
nq = length(idxQ);                  % Dimension of process noise
assert(length(xdw) == nx + nq);
x = xdw(1:nx);
dw = zeros(nx, 1);
dw(idxQ) = xdw(nx+1:end);

% Let \Delta t == n \delta t.
% Determine minimum of substeps required such that \delta t <= \delta t_{max}
nSubsteps = max(1, ceil(dt/obj.dtMaxEst));
dt = dt/nSubsteps;      % \Delta t = n \delta t
dw = dw/nSubsteps;      % \Delta w = n \delta w

t = obj.time;
if nargout < 2
    for j = 1:nSubsteps
        f1 = obj.dynamicsEst(t,        x);
        f2 = obj.dynamicsEst(t + dt/2, x + (f1*dt + dw)/2);
        f3 = obj.dynamicsEst(t + dt/2, x + (f2*dt + dw)/2);
        f4 = obj.dynamicsEst(t + dt,   x +  f3*dt + dw);
        x = x + (f1 + 2*f2 + 2*f3 + f4)*dt/6 + dw;
        t = t + dt;
    end
    xnext = x;
else
    % X(t)  = [ x(t),  dx(t)/dx[k],   dx(t)/dw[k] ]
    % dW(t) = [dw(t), ddw(t)/dx[k], ddw(t)/ddw[k] ]
    X = [x, eye(nx), zeros(nx)];    % X[k]  = [ x[k],  dx[k]/dx[k],   dx[k]/dw[k] ]
    dW = [dw, zeros(nx), eye(nx)];  % dW[k] = [dw[k], ddw[k]/dx[k], ddw[k]/ddw[k] ]
    for j = 1:nSubsteps
        F1 = obj.augmentedDynamicsEst(t,        X);
        F2 = obj.augmentedDynamicsEst(t + dt/2, X + (F1*dt + dW)/2);
        F3 = obj.augmentedDynamicsEst(t + dt/2, X + (F2*dt + dW)/2);
        F4 = obj.augmentedDynamicsEst(t + dt,   X +  F3*dt + dW);
        % X[k+1] = [ x[k+1], dx[k+1]/dx[k], dx[k+1]/dw[k] ]
        X = X + (F1 + 2*F2 + 2*F3 + F4)*dt/6 + dW;
        t = t + dt;
    end
    xnext = X(:, 1);
    Jdx = X(:, 2:1+nx);
    Jdw = X(:, 2+nx:end)/nSubsteps;
    % Since \Delta w = n \delta w, then 
    % \frac{\partial \mathbf{x}}{\partial \Delta\mathbf{w}} = \frac{\partial \mathbf{x}}{\partial \delta\mathbf{w}} \frac{1}{n}
    % therefore we divide Jdw by nSubsteps.
    J = [Jdx, Jdw(:, idxQ)];
end
