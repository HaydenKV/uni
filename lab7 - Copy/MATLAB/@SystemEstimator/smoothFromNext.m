function obj = smoothFromNext(obj, systemNext)

nx = obj.density.dim();
nxa = obj.densityAugmented.dim(); % [ x[k]; x(idxQ)[k+1] ]

idxA = 1:nx;
idxB = nx+1:nxa;

[~, idxQ] = systemNext.processNoise(1);
pxq = systemNext.density.marginal(idxQ);
% obj.density = obj.densityAugmented.conditional(idxA, idxB, systemNext.density);
obj.density = obj.densityAugmented.conditional(idxA, idxB, pxq);