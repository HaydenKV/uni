function systemNext = predict(obj, timeNext)

dt = timeNext - obj.time;
assert(dt >= 0);

% Call superclass predict method first
systemNext = predict@SystemEstimator(obj, timeNext);

if dt > 1e-14
    % Update simulator state
    options = odeset('MaxStep', 0.1);
    
    % Simulate system with input signal
    func = @(t, x) obj.dynamicsSim(t, x);
    [~, x_] = ode45(func, [obj.time timeNext], obj.x_sim, options);
    x_ = x_.';
    systemNext.x_sim = x_(:, end);
end