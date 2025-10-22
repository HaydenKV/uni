% Return h(x) and derivatives w.r.t. x
function [h, dhdx, d2hdx2] = predict(obj, x, system)

h = hypot(obj.r1, x(1) - obj.r2);

if nargout >= 2
    %               dh_i
    % dhdx(i, j) = ------
    %               dx_j
    dhdx = [(x(1) - obj.r2)/h(1), 0, 0];
end

if nargout >= 3
    %                     d^2 h_i
    % d2h2dx(i, j, k) = -----------
    %                    dx_j dx_k
    d2hdx2 = zeros(1, 3, 3);
    d2hdx2(1, 1, 1) = obj.r1^2/h(1)^3;
end
