classdef MeasurementRADAR < MeasurementGaussianLikelihood
    properties (Access = private, Constant)
        r1 = 5000;                  % Horizontal position of sensor [m]
        r2 = 5000;                  % Vertical position of sensor [m]
    end

    methods (Static)
        s = getProcessString()
    end

    methods
        [h, dhdx, d2hdx2] = predict(obj, x, system)     % Return h(x) and derivatives w.r.t. x
        out = noiseDensity(obj, system)                 % Return noise density
    end
end
