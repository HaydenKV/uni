function systemNext = predict(obj, timeNext)

dt = timeNext - obj.time;
assert(dt >= 0);

systemNext = obj;    % Copy system state (System has value semantics since it doesn't inherit from handle)

% Update time stamp
systemNext.time = timeNext;

% Update estimator state
if obj.runEstimator
    % Augment state density with independent noise increment dw ~ N(0, Q*dt)
    % [  x ] ~ N([ mu ], [ P,    0 ])
    % [ dw ]    ([  0 ]  [ 0, Q*dt ])
    [pdw, idxQ] = obj.processNoise(dt);
    pxdw = obj.density.join(pdw);   % p(x[k], dw(idxQ)[k]) = p(x[k])*p(dw(idxQ)[k])
    
    if obj.enableSmoother
        % Phi maps [ x[k]; dw(idxQ)[k] ] to [ x[k]; x(idxQ)[k+1] ]
        Phi = @(xdw) obj.RK4SDEHelperAugmented(xdw, dt, idxQ);

        % Propagate p(x[k], dw[k]) through RK4 to obtain p(x[k], x(idxQ)[k+1])
        systemNext.densityAugmented = pxdw.affineTransform(Phi);

        % Note: we can't recover the marginal p(x[k+1]) from the joint p(x[k], x(idxQ)[k+1])
        % % Obtain p(x[k+1]) from p(x[k], x[k+1])
        % nx = obj.density.dim();
        % % systemNext.density = systemNext.densityAugmented.marginal(nx+1:2*nx);
    end

    % Phi maps [ x[k]; dw(idxQ)[k] ] to x[k+1]
    Phi = @(xdw) obj.RK4SDEHelper(xdw, dt, idxQ);

    % Propagate p(x[k], dw[k]) through RK4 to obtain p(x[k+1])
    systemNext.density = pxdw.affineTransform(Phi);
end
