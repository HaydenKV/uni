function out = join(obj, other)

nuj = [obj.nu; other.nu];
Xij = blkdiag(obj.Xi, other.Xi);
out = GaussianInfo(nuj, Xij);
