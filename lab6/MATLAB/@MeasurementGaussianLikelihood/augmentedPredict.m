% Augmented measurement model
% [ y ] = [ h(x) + v ]
% [ x ]   [     x    ]
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
    Ja = [J, eye(ny, ny); eye(nx, nx), zeros(nx, ny)];
end
ya = [h + v; x];
