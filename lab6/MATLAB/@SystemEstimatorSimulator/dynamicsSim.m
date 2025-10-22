function [f, Jx, Ju] = dynamicsSim(obj, t, x)

% Default implementation reuses dynamics
u = obj.input(t, x);
switch nargout
    case 1
        f = obj.dynamics(t, x);
    case 2
        [f, Jx] = obj.dynamics(t, x);
    case 3
        [f, Jx, Ju] = obj.dynamics(t, x);
end
