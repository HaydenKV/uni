% Augmented measurement model
% [ x ]   [     x    ]
% [ y ] = [ h(x) + v ]
% \___/   \__________/
%   ya  =   ha(x, v)

% Return ya(x) and the Jacobian Ja = dha/dx
function [ya, Ja] = augmentedPredict(obj, xv, system)

nxv = length(xv);
ny = length(obj.y);
nx = nxv - ny;
x = xv(1:nx);
v = xv(nx+1:nx+ny);
if nargout < 2
    h = obj.predict(x, system);
else
    [h, J] = obj.predict(x, system);
    Ja = [eye(nx, nx), zeros(nx, ny); J, eye(ny, ny)];
end
ya = [x; h + v];
