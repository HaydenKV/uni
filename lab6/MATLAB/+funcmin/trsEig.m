% Trust-region subproblem (Eigendecomposition of Hessian)
% minimise 0.5*p.'*H*p + g.'*p subject to ||p|| < Delta
function [p, ret] = trsEig(Q, v, g, Delta)

arguments
    Q (:, :) double
    v (:, 1) double
    g (:, 1) double
    Delta (1, 1) double
end

assert(size(g, 2) == 1);
assert(size(v, 2) == 1);
assert(size(Q, 1) == size(Q, 2));
assert(size(Q, 1) == size(g, 1));
assert(size(v, 1) == size(g, 1));

sqrteps = realsqrt(eps);
maxIterations = 20;

l1 = min(v); % Leftmost eigenvalue
a = Q.'*g;

if l1 < 0
    lam = 1.01*abs(l1);
else
    lam = 0;
end

vlam = v + lam;
p = -Q*(a./vlam);

if l1 < 0 || norm(p) > Delta || abs(lam*(norm(p) - Delta)) > sqrteps
    isHardCase = abs(a(1)) < eps && l1 < 0;
    if isHardCase
        idxValid = find(abs(v - l1) > sqrteps);
        scaledValid = a(idxValid)./(v(idxValid) - l1);
        QValid = Q(:, idxValid);
        t = sqrt(Delta^2 - sum(scaledValid.^2));
        pvec = zeros(size(v));
        if ~isempty(idxValid)
            pvec = QValid*scaledValid;
        end
        p = t*Q(:, 1) - pvec;
        q = Q.'*p;
        if (p.'*g + 0.5*q.'*(v.*q) > 0)
            p = -t*Q(:, 1) - pvec;
        end
    else
        for k = 1:maxIterations
            pp = -a./vlam;
            dp = a./(vlam.^2);
            ppnorm = norm(pp);
            ff = 1/Delta - 1/ppnorm;
            gg = (dp.'*pp)/ppnorm^3;
            lam = max(max(0, -l1) + sqrteps*max(0, -l1), lam - ff/gg);
            vlam = v + lam;
            if abs(ff) < sqrteps
                break;
            end
        end
        p = -Q*(a./vlam);
        if k >= maxIterations
            ret = 1;
            return;
        end
    end
end

ret = 0;
