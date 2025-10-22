function out = add(obj, other)

assert(obj.dim() == other.dim());
mup = obj.mean() + other.mean();
Sp = qr([obj.sqrtCov(); other.sqrtCov()], "econ");
out = GaussianInfo.fromSqrtMoment(mup, Sp);
