function mu = mean(obj)

% Solve Xi*mu = nu for mu
mu = linsolve(obj.Xi, obj.nu, obj.s_ut); % Xi\nu
