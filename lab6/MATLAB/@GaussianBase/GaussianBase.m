classdef GaussianBase < Density
    properties (Constant, Access=protected)
        % static const structures for fast triangular solves
        s_ut = struct('UT', true, 'TRANSA', false);         % for solving R*x = b for x with upper triangular R (triangular backward substitution)
        s_ut_transa = struct('UT', true, 'TRANSA', true);   % for solving R.'*x = b for x with upper triangular R (triangular forward substitution)
    end

    % Factory methods to construct a Gaussian from given parameters
    methods (Static, Abstract)
        out = fromSqrtMoment(varargin)  % Construct a Gaussian from mean (optional) and sqrt covariance
        out = fromMoment(varargin)      % Construct a Gaussian from mean (optional) and covariance
        out = fromSqrtInfo(varargin)    % Construct a Gaussian from sqrt information vector (optional) and sqrt information matrix
        out = fromInfo(varargin)        % Construct a Gaussian from information vector (optional) and information matrix
        out = fromSamples(X)            % Construct a Gaussian from a set of samples
    end

    methods (Access=protected)
        % Protected constructor for internal use
        function obj = GaussianBase()
            % Call superclass constructor(s)
            obj@Density();
        end
    end

    methods (Abstract)
        n = dim(obj)                                    % Return dimension of Gaussian
        mu = mean(obj)                                  % Return mean vector
        S = sqrtCov(obj)                                % Return upper-triangular square-root covariance matrix
        P = cov(obj)                                    % Return covariance matrix
        eta = infoVec(obj)                              % Return information vctor
        Xi = sqrtInfoMat(obj)                           % Return upper-triangular square-root information matrix
        nu = sqrtInfoVec(obj)                           % Return square-root information vector
        Lambda = infoMat(obj)                           % Return information matrix
        out = join(obj, other)                          % Construct joint Gaussian from product of independent marginals, p(x1, x2) = p(x1)*p(x2)
        out = marginal(obj, idx)                        % Given joint density p(x), return marginal density p(x(idx))
        out = conditional(obj, idxA, idxB, xB)          % Given joint density p(x), return conditional density p(x(idxA) | x(idxB) = xB)
        [logPDF, g, H] = log(obj, X)                    % Log likelihood evaluated at columns of X and optional gradient and Hessian w.r.t. X
        b = isWithinConfidenceRegion(obj, x, nSigma)    % Test if x is within nSigma standard deviations
        Q = quadricSurface(obj, nSigma)                 % Quadric surface coefficients for a given number of standard deviations
    end

    methods
        X = simulate(obj, N)                            % Simulate N realisations
        [l, dldmu] = logIntegral(obj, a, b)             % Compute log int_a^b N(x; mu, sigma^2) dx and its derivative w.r.t. mu
        X = confidenceEllipse(obj, nSigma, nSamples)    % Points on boundary of confidence ellipse for a given number of standard deviations
    end
end
