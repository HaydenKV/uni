function [f, Jx] = dynamicsEst(obj, t, x)

% Default implementation reuses dynamics
u = obj.input(t, x);
switch nargout
    case 1
        f = obj.dynamics(t, x, u);
    case 2
        [f, Jx] = obj.dynamics(t, x, u);
end