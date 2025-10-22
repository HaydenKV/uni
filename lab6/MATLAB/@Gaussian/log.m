function [logPDF, g, H] = log(obj, X)

arguments (Input)
    obj (1, 1) Gaussian
    X (:, :) double                 % n-by-m matrix of points to evaluate log at, where n is the dimension and m is the number of evaluation points
end

arguments (Output)
    logPDF (1, :) double            % log of Gaussian at evaluation points (1-by-m)
    g (:, :) double                 % gradient of logPDF at evaluation points (n-by-m)
    H (:, :, :) double              % Hessian of logPDF at evaluation points (n-by-n-by-m)
end

[n, m] = size(X);
assert(n == obj.dim());

Z = linsolve(obj.S, X - obj.mu, obj.s_ut_transa); % S.'\(X - mu); (triangular forward substitution)
logPDF = nan(1, m);
% TODO: Merge from MCHA4100

if nargout >= 2
    % Compute gradient g = -S\(S.'\(X - mu))
    g = -linsolve(obj.S, Z, obj.s_ut); % g = -S\Z; (triangular backward substitution)
end

if nargout >= 3
    % Compute Hessian H = -S\(S.'\eye(n))
    H0 = -obj.infoMat();       % Reuse existing implementation since H = -Lambda

    % Since Hessian is the same for any point x, just replicate it for each input
    H = repmat(H0, [1, 1, m]);
end
