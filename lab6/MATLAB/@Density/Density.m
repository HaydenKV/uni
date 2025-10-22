classdef Density
    methods (Abstract)
        logPDF = log(obj, X)                            % Log likelihood evaluated at columns of X
    end

    methods
        pdf = eval(obj, X)                              % Evaluate pdf at columns of X
    end
end