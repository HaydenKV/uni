function [l, dldx, d2ldx2] = logLikelihood(obj, x, system)

switch nargout
    case 1
        likelihood = obj.predictDensity(x, system);
        l = likelihood.log(obj.y);
    case 2
        [likelihood, dhdx] = obj.predictDensity(x, system);
        [l, dldy] = likelihood.log(obj.y);

        % Gradient of log likelihood:
        %
        %         d
        % g_i = ---- log N(y; h(x), R)
        %       dx_i
        %
        %             dh_k     d
        % g_i = sum_k ---- * ---- log N(y; h(x), R)
        %             dx_i   dh_k
        %
        %               dh_k     d
        % g_i = - sum_k ---- * ---- log N(y; h(x), R)
        %               dx_i   dy_k
        %
        dldx = -dhdx.'*dldy;
    case 3
        [likelihood, dhdx, d2hdx2] = obj.predictDensity(x, system);
        [l, dldy, d2ldy2] = likelihood.log(obj.y);
        dldx = -dhdx.'*dldy;

        % Hessian of log likelihood:
        %
        %              d                                 d  ( dh_k     d                    )
        % H_{ij} = --------- log N(y; h(x), R) = sum_k ---- ( ---- * ---- log N(y; h(x), R) )
        %          dx_i dx_j                           dx_j ( dx_i   dh_k                   )
        %
        %                      dh_k   d^2 log N(y; h(x), R)   dh_l          d^2 h_k      d
        % H_{ij} = sum_k sum_l ---- * --------------------- * ---- + sum_k --------- * ---- log N(y; h(x), R)
        %                      dx_i         dh_k dh_l         dx_j         dx_i dx_j   dh_k
        %
        %                      dh_k   d^2 log N(y; h(x), R)   dh_l          d^2 h_k      d
        % H_{ij} = sum_k sum_l ---- * --------------------- * ---- - sum_k --------- * ---- log N(y; h(x), R)
        %                      dx_i         dy_k dy_l         dx_j         dx_i dx_j   dy_k
        %
        nh = length(obj.y);
        nx = length(x);
        d2ldx2 = dhdx.'*d2ldy2*dhdx - reshape(sum(d2hdx2 .* reshape(dldy, [nh, 1, 1]), 1), [nx, nx]);
end

