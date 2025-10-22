function out = fromMoment(varargin)

if nargin == 2
    % If there are two arguments, compute the Cholesky decomposition of the second one
    varargin{2} = chol(varargin{2});
end
out = Gaussian(varargin{:});
