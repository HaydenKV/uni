function [pdw, idx] = processNoise(obj, dt)

% Square-root power spectral density of continuous-time process noise
SQ = diag([1e-10, 5e-6]);

% Indices of process model equations where process noise is injected
idx = [2, 3];

% Distribution of noise increment dw ~ N(0, Q*dt) for time increment dt
pdw = Gaussian.fromSqrtMoment(SQ*realsqrt(dt));

