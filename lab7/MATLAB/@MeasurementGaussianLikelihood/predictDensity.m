% Return density p(y|x) for a given x and derivatives of mean w.r.t. x
function [py, dhdx, d2hdx2] = predictDensity(obj, x, system)

switch nargout
    case 1
        h = obj.predict(x, system);
    case 2
        [h, dhdx] = obj.predict(x, system);
    case 3
        [h, dhdx, d2hdx2] = obj.predict(x, system);
end

Xi = obj.noiseDensity(system).sqrtInfoMat();
py = GaussianInfo.fromSqrtInfo(Xi*h, Xi);
