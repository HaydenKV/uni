function [obj, system] = update(obj, system)

if obj.needToSimulate
    obj = obj.simulate(system.x_sim, system);
    obj.needToSimulate = false;
end

switch obj.updateMethod
    case 'affine'
        pxv = system.density.join(obj.noiseDensity(system)); % p(x, v) = p(x)*p(v)
        jointFunc = @(x) obj.augmentedPredict(x, system);
        pxy = pxv.affineTransform(jointFunc);
        nx = system.density.dim();
        ny = length(obj.y);
        idxX = 1:nx;
        idxY = nx+1:nx+ny;
        system.density = pxy.conditional(idxX, idxY, obj.y);
    case 'gaussnewton'
        error('Not yet implemented');
    case 'levenbergmarquardt'
        error('Not yet implemented');
    otherwise
        [obj, system] = obj.update@Measurement(system);
end
