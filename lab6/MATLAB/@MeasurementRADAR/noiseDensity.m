function out = noiseDensity(obj, system)

S_RADAR = 50;

out = Gaussian.fromSqrtMoment(S_RADAR);
