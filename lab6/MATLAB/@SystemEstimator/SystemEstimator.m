classdef SystemEstimator < SystemBase
    properties
        density Gaussian                        % Estimator state
        runEstimator (1, 1) logical = true      % Run state estimator
        dtMaxEst (1, 1) double = 1e-2;          % Maximum time step for process model prediction
    end

    methods
        [f, Jx] = dynamicsEst(obj, t, x)
        F = augmentedDynamicsEst(obj, t, X)
        [xnext, J] = RK4SDEHelper(obj, xdw, dt, idxQ)
        systemNext = predict(obj, timeNext)
    end

    methods (Abstract)
        [pdw, idx] = processNoise(obj, dt)
    end
end
