function [obj, system] = update(obj, system)

if obj.needToSimulate
    obj = obj.simulate(system.x_sim, system);
    obj.needToSimulate = false;
end

x = system.density.mean();          % Set initial decision variable to prior mean
Xi = system.density.sqrtInfoMat();  % Set initial sqrt Hessian to prior sqrt info matrix

costFunc = @(x) obj.costJointDensity(x, system);

switch obj.updateMethod
    case 'BFGSTrustSqrt'
        [x, ~, Xi, ret] = funcmin.BFGSTrustSqrt(costFunc, x, Xi, obj.verbosity);
        assert(ret == 0);
    case 'BFGSLMSqrt'
        [x, ~, Xi, ret] = funcmin.BFGSLMSqrt(costFunc, x, Xi, obj.verbosity);
        assert(ret == 0);
    case 'SR1TrustEig'
        % Generate eigendecomposition of initial Hessian (prior information matrix)
        % via an SVD of Xi = U*D*V.', i.e., Xi.'*Xi = V*D*U.'*U*D*V.' = V*D^2*V.'
        % This avoids the loss of precision associated with directly computing the eigendecomposition of Xi.'*Xi
        [~, d, Q] = svd(Xi, 'vector');
        v = d.^2;

        % Foreshadowing for MCHA4400:
        %   If we were doing landmark SLAM with a quasi-Newton method,
        %   we can purposely introduce negative eigenvalues for newly
        %   initialised landmarks to force the Hessian and hence
        %   posterior sqrt information matrix to be approximated correctly.

        [x, ~, Q, v, ret] = funcmin.SR1TrustEig(costFunc, x, Q, v, obj.verbosity);
        assert(ret == 0);

        % Post-calculate posterior square-root information matrix from Hessian eigendecomposition
        Xi = qr(Q.'.*realsqrt(v), "econ");
    case 'NewtonTrustEig'
        [x, ~, Q, v, ret] = funcmin.NewtonTrustEig(costFunc, x, obj.verbosity);
        assert(ret == 0);

        % Post-calculate posterior square-root information matrix from Hessian eigendecomposition
        Xi = qr(Q.'.*realsqrt(v), "econ");
    otherwise
        error('Unsupported update method');
end

% Update posterior density
mu = x;     % Set posterior mean to maximum a posteriori (MAP) estimate
system.density = GaussianInfo.fromSqrtInfo(Xi*mu, Xi);
