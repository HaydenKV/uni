function [l, dldmu] = logIntegral(obj, a, b)

% Compute log int_a^b N(x; mu, sigma^2) dx and its derivative w.r.t. mu

assert(obj.dim() == 1, 'Expected univariate Gaussian to compute the log-integral')
assert(b >= a, 'Expected integral to be non-negative so that the log exists')

mu = obj.mean();
sigma = obj.sqrtCov();

za = -1/realsqrt(2)*(a - mu)/sigma;
zb = -1/realsqrt(2)*(b - mu)/sigma;
ta = reallog(0.5*erfcx(abs(za))) - za^2;
tb = reallog(0.5*erfcx(abs(zb))) - zb^2;

if za >= 0
    if zb >= 0
        l = tb + log1p(-exp(ta - tb));
    else % zb < 0
        l = log1p(-exp(ta) - exp(tb));
    end
else % za < 0
    if zb >= 0
        error('Case forbidden, since b >= a implies 0 <= zb <= za < 0');
    else % zb < 0
        l = ta + log1p(-exp(tb - ta));
    end
end

if nargout >= 2
    dldmu = exp(-za^2 - l - 0.5*reallog(2*pi*sigma^2)) - exp(-zb^2 - l - 0.5*reallog(2*pi*sigma^2));
end