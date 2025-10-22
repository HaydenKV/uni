classdef SystemBase < matlab.mixin.Heterogeneous
    properties
        time (1, 1) double = 0                  % Time associated with current system state
    end
    
    methods (Abstract)
        [f, Jx, Ju] = dynamics(obj, t, x, u)    % Used by simulator, estimator, controller
        u = input(obj, t, x)
        systemNext = predict(obj, time)         % Called by an event to predict the system state forward to a given time
    end
end

