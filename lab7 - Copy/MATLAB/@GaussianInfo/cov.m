function P = cov(obj)

S = obj.sqrtCov();
P = S.'*S;
