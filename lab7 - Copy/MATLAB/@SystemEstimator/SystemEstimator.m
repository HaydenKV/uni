classdef SystemEstimator < SystemBase
    properties
        density GaussianInfo                    % Estimator state
        runEstimator (1, 1) logical = true      % Run state estimator
        enableSmoother (1, 1) logical = false           % Save extra augmented density needed for later smoothing pass
        densityAugmented GaussianInfo = GaussianInfo.empty  % Augmented density needed for later smoothing pass
        dtMaxEst (1, 1) double = 1e-2;          % Maximum time step for process model prediction
    end

    methods
        [f, Jx] = dynamicsEst(obj, t, x)
        F = augmentedDynamicsEst(obj, t, X)
        [xnext, J] = RK4SDEHelper(obj, xdw, dt, idxQ)
        [xa, Ja] = RK4SDEHelperAugmented(obj, xdw, dt, idxQ)
        systemNext = predict(obj, timeNext)
        obj = smoothFromNext(obj, systemNext)
    end

    methods (Abstract)
        [pdw, idx] = processNoise(obj, dt)
    end
end
