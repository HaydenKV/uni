function out = affineTransform(obj, h)

[muy, J] = h(obj.mu);   % Evaluate function at mean value
Syy = qr(obj.S*J.', "econ");
out = Gaussian(muy, Syy);
