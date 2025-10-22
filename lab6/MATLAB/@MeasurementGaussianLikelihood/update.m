function [obj, system] = update(obj, system)

if obj.needToSimulate
    obj = obj.simulate(system.x_sim, system);
    obj.needToSimulate = false;
end

switch obj.updateMethod
    case {'affine', 'unscented'}
        transform = [obj.updateMethod 'Transform'];
        pxv = system.density.join(obj.noiseDensity(system)); % p(x, v) = p(x)*p(v)
        jointFunc = @(x) obj.augmentedPredict(x, system);
        pyx = pxv.(transform)(jointFunc);
        nx = system.density.dim();
        ny = length(obj.y);
        idxX = ny+1:nx+ny;
        idxY = 1:ny;
        system.density = pyx.conditional(idxX, idxY, obj.y);
    case 'gaussnewton'
        error('Not yet implemented');
    case 'levenbergmarquardt'
        error('Not yet implemented');
    otherwise
        [obj, system] = obj.update@Measurement(system);
end
