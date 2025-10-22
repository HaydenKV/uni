classdef Measurement < Event
    properties
        updateMethod = 'NewtonTrustEig'
        needToSimulate(1, 1) logical = false
    end
    
    methods (Abstract)
        obj = simulate(obj, x, system)
        [l, g, H] = logLikelihood(obj, x, system)
    end

    methods (Access = protected)
        [V, g, H] = costJointDensity(obj, x, system)
        [obj, system] = update(obj, system)
    end
end