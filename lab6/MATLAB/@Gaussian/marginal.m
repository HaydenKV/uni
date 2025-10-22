function out = marginal(obj, idx)

mum = obj.mu(idx);
Sm = qr(obj.S(:, idx), "econ");
out = Gaussian(mum, Sm);
