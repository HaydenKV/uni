function out = noiseDensity(obj, system)

S_RADAR = 50;

out = GaussianInfo.fromSqrtMoment(S_RADAR);
