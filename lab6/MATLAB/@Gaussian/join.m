function out = join(obj, other)

muj = [obj.mu; other.mu];
Sj = blkdiag(obj.S, other.S);
out = Gaussian(muj, Sj);
