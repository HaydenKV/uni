function [logPDF, g, H] = log(obj, X)

%
% Input arguments
%
% obj:      GaussianInfo object
% X:        points to evaluate (n x m)
%
% where
%   n is the dimension
%   m is the number of evaluation points
%
% Output arguments
%
% logPDF:   log of Gaussian at evaluation points (1 x m)
% g:        gradient of logPDF at evaluation points (n x m)
% H:        Hessian of logPDF at evaluation points (n x n x m)
%

[n, m] = size(X);
assert(n == obj.dim());

Z = obj.Xi*X - obj.nu;
logPDF = -0.5*sum(Z.^2, 1) + sum(reallog(abs(diag(obj.Xi)))) - n/2*reallog(2*pi);

if nargout >= 2
    % Compute gradient g = -Xi.'*(Xi*X - nu)
    g = -obj.Xi.'*Z;
end

if nargout >= 3
    % Compute Hessian H = -Xi.'*Xi
    H0 = -obj.infoMat();       % Reuse existing implementation since H = -Lambda

    % Since Hessian is the same for any point x, just replicate it for each input
    H = repmat(H0, [1, 1, m]);
end
