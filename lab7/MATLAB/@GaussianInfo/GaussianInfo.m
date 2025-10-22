classdef GaussianInfo < GaussianBase
    properties (SetAccess=protected)
        nu (:, 1) double
        Xi (:, :) double
    end

    methods (Access=protected)
        % Protected constructor for internal use
        function obj = GaussianInfo(arg1, arg2)
            % Call superclass constructor(s)
            obj@GaussianBase();

            switch nargin
                case 0      % Construct empty GaussianInfo
                    obj.nu = zeros(0, 1);
                    obj.Xi = zeros(0, 0);
                case 1      % Construct zero-sqrt-info-vector GaussianInfo with given sqrt info matrix
                    obj.Xi = arg1;
                    obj.nu = zeros(size(obj.Xi, 2), 1);
                case 2      % Construct GaussianInfo from sqrt info vector and sqrt info matrix
                    obj.nu = arg1;
                    obj.Xi = arg2;
            end
            assert(istriu(obj.Xi), 'Expected Xi to be upper triangular');
            assert(iscolumn(obj.nu), 'Expected nu to be a column vector');
            assert(length(obj.nu) == size(obj.Xi, 1), 'Expected dimensions of nu and Xi to be compatible');
        end
    end

    % Factory methods to construct a GaussianInfo from given parameters
    methods (Static)
        out = fromSqrtMoment(varargin)  % Construct a GaussianInfo from mean (optional) and sqrt covariance
        out = fromMoment(varargin)      % Construct a GaussianInfo from mean (optional) and covariance
        out = fromSqrtInfo(varargin)    % Construct a GaussianInfo from sqrt information vector (optional) and sqrt information matrix
        out = fromInfo(varargin)        % Construct a GaussianInfo from information vector (optional) and information matrix
        out = fromSamples(X)            % Construct a GaussianInfo from a set of samples
    end

    methods
        n = dim(obj)                                    % Return dimension of GaussianInfo
        mu = mean(obj)                                  % Return mean vector
        S = sqrtCov(obj)                                % Return upper-triangular square-root covariance matrix
        P = cov(obj)                                    % Return covariance matrix
        eta = infoVec(obj)                              % Return information vctor
        Xi = sqrtInfoMat(obj)                           % Return upper-triangular square-root information matrix
        nu = sqrtInfoVec(obj)                           % Return square-root information vector
        Lambda = infoMat(obj)                           % Return information matrix
        out = join(obj, other)                          % Construct joint GaussianInfo from product of independent marginals, p(x1, x2) = p(x1)*p(x2)
        out = marginal(obj, idx)                        % Given joint density p(x), return marginal density p(x(idx))
        out = conditional(obj, idxA, idxB, xB)          % Given joint density p(x), return conditional density p(x(idxA) | x(idxB) = xB)
        out = affineTransform(obj, h)                   % Affine transform of y = h(x)
        [logPDF, g, H] = log(obj, X)                    % Log likelihood evaluated at columns of X and optional gradient and Hessian w.r.t. X
        b = isWithinConfidenceRegion(obj, x, n_sigma)   % Test if x is within n_sigma standard deviations
        Q = quadricSurface(obj, nSigma)                 % Quadric surface coefficients for a given number of standard deviations
    end
end
