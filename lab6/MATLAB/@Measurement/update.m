function [obj, system] = update(obj, system)

if obj.needToSimulate
    obj = obj.simulate(system.x_sim, system);
    obj.needToSimulate = false;
end

x = system.density.mean();      % Set initial decision variable to prior mean
S = system.density.sqrtCov();

costFunc = @(x) obj.costJointDensity(x, system);

switch obj.updateMethod
    case 'BFGSTrustSqrtInv'
        [x, ~, S, ret] = funcmin.BFGSTrustSqrtInv(costFunc, x, S, obj.verbosity);
        assert(ret == 0);
    case 'SR1TrustEig'
        % Generate eigendecomposition of initial Hessian (inverse of prior covariance)
        % via an SVD of S = U*D*V.', i.e., (S.'*S)^{-1} = (V*D*U.'*U*D*V.')^{-1} = V*D^{-2}*V.'
        % This avoids the loss of precision associated with directly computing the eigendecomposition of (S.'*S)^{-1}
        [~, d, Q] = svd(S, 'vector');
        v = 1./d.^2;

        % Foreshadowing for MCHA4400:
        %   If we were doing landmark SLAM with a quasi-Newton method,
        %   we can purposely introduce negative eigenvalues for newly
        %   initialised landmarks to force the Hessian and hence
        %   posterior covariance to be approximated correctly.

        [x, ~, Q, v, ret] = funcmin.SR1TrustEig(costFunc, x, Q, v, obj.verbosity);
        assert(ret == 0);

        % Post-calculate posterior square-root covariance from Hessian eigendecomposition
        S = qr(Q.'./realsqrt(v), "econ");
    case 'NewtonTrustEig'
        [x, ~, Q, v, ret] = funcmin.NewtonTrustEig(costFunc, x, obj.verbosity);
        assert(ret == 0);

        % Post-calculate posterior square-root covariance from Hessian eigendecomposition
        S = qr(Q.'./realsqrt(v), "econ");
    otherwise
        error('Unsupported update method');
end

% Update posterior density
mu = x;     % Set posterior mean to maximum a posteriori (MAP) estimate
system.density = Gaussian.fromSqrtMoment(mu, S);
