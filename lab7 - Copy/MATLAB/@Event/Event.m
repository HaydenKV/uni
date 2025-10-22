classdef Event < matlab.mixin.Heterogeneous
    properties
        time (1, 1) double
        saveSystemState (1, 1) logical = true
        system SystemBase = SystemBase.empty
        verbosity (1, 1) uint8 = 1
    end

    methods (Access = protected)
        [obj, system] = update(obj, system)
    end

    methods (Static)
        s = getProcessString()
    end
    
    methods
        [obj, system] = process(obj, system)
    end
    
    methods (Sealed)
        obj = sort(obj)
        obj = smooth(obj)
    end
end
