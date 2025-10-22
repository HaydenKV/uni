function obj = simulate(obj, x, system)

obj.y = obj.predictDensity(x, system).simulate();
