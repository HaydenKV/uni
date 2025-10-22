function [V, g, H] = costJointDensity(obj, x, system)

switch nargout
    case 1
        logprior = system.density.log(x);
        loglik = obj.logLikelihood(x, system);
        V = -(logprior + loglik);
    case 2
        [logprior, logpriorGrad] = system.density.log(x);
        [loglik, loglikGrad] = obj.logLikelihood(x, system);
        V = -(logprior + loglik);
        g = -(logpriorGrad + loglikGrad);
    case 3
        [logprior, logpriorGrad, logpriorHess] = system.density.log(x);
        [loglik, loglikGrad, loglikHess] = obj.logLikelihood(x, system);
        V = -(logprior + loglik);
        g = -(logpriorGrad + loglikGrad);
        H = -(logpriorHess + loglikHess);
end
