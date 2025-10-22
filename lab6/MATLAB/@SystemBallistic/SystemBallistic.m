classdef SystemBallistic < SystemEstimatorSimulator
    properties (Constant)
        p0 = 101.325e3;             % Air pressure at sea level [Pa]
        M  = 0.0289644;             % Molar mass of dry air [kg/mol]
        R  = 8.31447;               % Gas constant [J/(mol.K)]
        L  = 0.0065;                % Temperature gradient [K/m]
        T0 = 288.15;                % Temperature at sea level [K]
        g  = 9.81;                  % Acceleration due to gravity [m/s^2]
    end

    methods
        function obj = SystemBallistic()
            % Call superclass constructor(s)
            obj@SystemEstimatorSimulator();

            % Initial time
            obj.time = 0;

            % Initial simulator state
            obj.x_sim = [ ...
                15000; ...          % Initial height
                -500; ...           % Initial velocity
                0.001 ...           % Ballistic coefficient
                ];

            % Initial estimator state
            mu0 = [ ...
                14000; ...          % Initial height
                -450; ...           % Initial velocity
                0.0005 ...          % Ballistic coefficient
                ];
            S0 = diag([2200, 100, 1e-3]);
            obj.density = Gaussian.fromSqrtMoment(mu0, S0);
        end

        [f, J] = dynamics(obj, t, x, u)
        u = input(obj, t, x)
        [pdw, idx] = processNoise(obj, dt)
    end
end