classdef MeasurementGaussianLikelihood < Measurement
    properties
        y (:, 1) double         % Measurement vector
    end
    
    methods (Abstract)
        [h, dhdx, d2hdx2] = predict(obj, x, system)     % Return h(x) and derivatives w.r.t. x
        out = noiseDensity(obj, system)                 % Return p(v)
    end

    methods
        function obj = MeasurementGaussianLikelihood()
            obj = obj@Measurement();
            obj.updateMethod = 'affine';
        end

        obj = simulate(obj, x, system)
        [l, dldx, d2ldx2] = logLikelihood(obj, x, system)
        [py, dhdx, d2hdx2] = predictDensity(obj, x, system)
    end
    
    methods (Access = protected)
        [ya, Ja] = augmentedPredict(obj, xv, system)      % Return ya(x) and the Jacobian Ja = dha/dx
    end
    
    methods (Access = protected)
        [obj, system] = update(obj, system)
    end
end
