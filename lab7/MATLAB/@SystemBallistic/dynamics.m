function [f, J] = dynamics(obj, t, x, u)

T = obj.T0 - obj.L*x(1);                                        % Air temperature at current altitude
gM_RL = obj.g*obj.M/obj.R/obj.L;
T_T0 = T/obj.T0;
p = obj.p0*T_T0^gM_RL;                                          % Air pressure at current altitude
rho = p*obj.M/obj.R/T;                                          % Air density at current altitude

d = 1/2*rho*x(2)^2*x(3);                                        % Drag acceleration

% Evaluate f(x) from dx = f(x)*dt + dw
f = [ ...
    x(2); ...           % Kinematic relation
    d - obj.g; ...      % Acceleration experienced by falling body
    0 ...               % Ballistic coefficient is constant
    ];

if nargout >= 2
    % Jacobian matrix J = df/dx
    J = [0, 1, 0; ...
        (obj.M*obj.p0*obj.L*obj.R*(1 - gM_RL)*T_T0^gM_RL*x(2)^2*x(3))/(2*obj.R^2*T^2), ...
        (obj.M*obj.p0*T_T0^gM_RL*x(2)*x(3))/(obj.R*T), ...
        (obj.M*obj.p0*T_T0^gM_RL*x(2)^2)/(2*obj.R*T); ...
        0, 0, 0];
end
