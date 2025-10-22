function out = add(obj, other)

n = obj.dim();
assert(n == other.dim());
mup = obj.mu + other.mu;
Sp = qr([obj.S; other.S], "econ");
out = Gaussian(mup, Sp);
