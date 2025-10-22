classdef Gaussian < GaussianBase
    properties (SetAccess=protected)
        mu (:, 1) double
        S (:, :) double
    end

    methods (Access=protected)
        % Protected constructor for internal use
        function obj = Gaussian(arg1, arg2)
            % Call superclass constructor(s)
            obj@GaussianBase();

            switch nargin
                case 0      % Construct empty Gaussian
                    obj.mu = zeros(0, 1);
                    obj.S = zeros(0, 0);
                case 1      % Construct zero-mean Gaussian with given sqrt cov
                    obj.S = arg1;
                    obj.mu = zeros(size(obj.S, 2), 1);
                case 2      % Construct Gaussian from mean and sqrt cov
                    obj.mu = arg1;
                    obj.S = arg2;
            end
            assert(istriu(obj.S), 'Expected S to be upper triangular');
            assert(iscolumn(obj.mu), 'Expected mu to be a column vector')
            assert(length(obj.mu) == size(obj.S, 2), 'Expected dimensions of mu and S to be compatible');
        end
    end

    % Factory methods to construct a Gaussian from given parameters
    methods (Static)
        out = fromSqrtMoment(varargin)  % Construct a Gaussian from mean (optional) and sqrt covariance
        out = fromMoment(varargin)      % Construct a Gaussian from mean (optional) and covariance
        out = fromSqrtInfo(varargin)    % Construct a Gaussian from sqrt information vector (optional) and sqrt information matrix
        out = fromInfo(varargin)        % Construct a Gaussian from information vector (optional) and information matrix
        out = fromSamples(X)            % Construct a Gaussian from a set of samples
    end

    methods
        n = dim(obj)                                    % Return dimension of Gaussian
        mu = mean(obj)                                  % Return mean vector
        S = sqrtCov(obj)                                % Return upper-triangular square-root covariance matrix
        P = cov(obj)                                    % Return covariance matrix
        eta = infoVec(obj)                              % Return information vctor
        Xi = sqrtInfoMat(obj)                           % Return upper-triangular square-root information matrix
        nu = sqrtInfoVec(obj)                           % Return square-root information vector
        Lambda = infoMat(obj)                           % Return information matrix
        out = add(obj, other)                           % Given p(x1) and p(x2), return p(x1 + x2)
        out = join(obj, other)                          % Construct joint Gaussian from product of independent marginals, p(x1, x2) = p(x1)*p(x2)
        out = marginal(obj, idx)                        % Given joint density p(x), return marginal density p(x(idx))
        out = conditional(obj, idxA, idxB, dataB)       % Given joint density p(x), return conditional density p(x(idxA) | x(idxB) = xB) or p(x(idxA) | p(x(idxB)) = p(xB))
        out = affineTransform(obj, h)                   % Affine transform of y = h(x)
        out = unscentedTransform(obj, h)                % Unscented transform of y = h(x)
        [out, Y, X] = monteCarloTransform(obj, h, n_samples) % Monte-Carlo transform of y = h(x), and optional samples Y and X
        [logPDF, g, H] = log(obj, X)                    % Log likelihood evaluated at columns of X and optional gradient and Hessian w.r.t. X
        b = isWithinConfidenceRegion(obj, x, n_sigma)   % Test if x is within n_sigma standard deviations
        Q = quadricSurface(obj, n_sigma)                % Quadric surface coefficients for a given number of standard deviations
    end
end
