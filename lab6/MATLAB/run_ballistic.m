% Ensure no unit tests fail before continuing
results = runtests('tests');
assert(~any([results.Failed]));

%% Create event queue

% Note: In this problem, the event queue can be entire sequence since we
%       know all the event times in advance and the measurements are simulated

event_queue = Event.empty;

% % Create dummy events to save the system state at extra time steps
% for t = 0:0.02:60       % Event time stamps
%     dummy = Event();
%     dummy.time = t;
%     event_queue(end + 1) = dummy; %#ok<SAGROW>
% end

% Create measurement events
for t = 0.1:0.1:60      % Event time stamps
    measurement = MeasurementRADAR();
    measurement.time = t;
    % measurement.y = y;                      % Use given data
    measurement.needToSimulate = true;      % Use simulated data (relies on simulated state)
    measurement.updateMethod = 'BFGSTrustSqrtInv'; % 'affine', 'unscented', 'BFGSTrustSqrtInv', 'SR1TrustEig', 'NewtonTrustEig'
    % measurement.verbosity = 3;
    event_queue(end + 1) = measurement; %#ok<SAGROW>
end

% Sort event queue in non-decreasing time order
event_queue = sort(event_queue);

%% Create initial system and run event loop

% Create system
system = SystemBallistic();

% Run event loop
s = rng;    % Save random seed
rng(42);    % Set random seed
for k = 1:length(event_queue)
    [event_queue(k), system] = event_queue(k).process(system);  % Process event
end
rng(s);     % Restore random seed

%% Post-processing

% Get data for plotting from saved system state at each event
t_hist = nan(1, length(event_queue));
x_hist = nan(3, length(event_queue));
mu_hist = nan(3, length(event_queue));
sigma_hist = nan(3, length(event_queue));
y_hist = nan(1, length(event_queue));
for k = 1:length(event_queue)
    t_hist(:, k) = event_queue(k).time;
    if event_queue(k).saveSystemState
        x_hist(:, k) = event_queue(k).system.x_sim;
        mu_hist(:, k) = event_queue(k).system.density.mean();
        sigma_hist(:, k) = realsqrt(diag(event_queue(k).system.density.cov())); % Square root of diagonal of P
    end
    if isa(event_queue(k), 'MeasurementGaussianLikelihood')
        y_hist(:, k) = event_queue(k).y;
    end
end

n_sigma = 3;
mu_hist_m = mu_hist - n_sigma*sigma_hist;
mu_hist_p = mu_hist + n_sigma*sigma_hist;

%% Time-series plot
fig = 1;
hf = figure(fig); clf(fig);
hf.Position = [100, 100, 2*560, 2*420];
ax1 = subplot(3, 2, 1, 'Parent', hf);
title(ax1, 'State estimates', 'interpreter', 'latex', 'FontSize', 16)
hold(ax1, 'on')
plot(ax1, t_hist, x_hist(1, :), 'r')
plot(ax1, t_hist, mu_hist(1, :), 'b')
fill(ax1, [t_hist, fliplr(t_hist)], [mu_hist_m(1, :), fliplr(mu_hist_p(1, :))], 'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
hold(ax1, 'off')
xlabel(ax1, 'Time [s]')
ylabel(ax1, 'Height [m]')
legend(ax1, {'$x_1$ (true)', '$\mu_1$', '$\mu_1 \pm 3\sigma_1$'}, 'interpreter', 'latex')
grid(ax1, 'on')

ax3 = subplot(3, 2, 3, 'Parent', hf);
hold(ax3, 'on')
plot(ax3, t_hist, x_hist(2, :), 'r')
plot(ax3, t_hist, mu_hist(2, :), 'b')
fill(ax3, [t_hist, fliplr(t_hist)], [mu_hist_m(2, :), fliplr(mu_hist_p(2, :))], 'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
hold(ax3, 'off')
xlabel(ax3, 'Time [s]')
ylabel(ax3, 'Velocity [m/s]')
legend(ax3, {'$x_2$ (true)', '$\mu_2$', '$\mu_2 \pm 3\sigma_2$'}, 'interpreter', 'latex')
grid(ax3, 'on')

ax5 = subplot(3, 2, 5, 'Parent', hf);
hold(ax5, 'on')
plot(ax5, t_hist, x_hist(3, :), 'r')
plot(ax5, t_hist, mu_hist(3, :), 'b')
fill(ax5, [t_hist, fliplr(t_hist)], [mu_hist_m(3, :), fliplr(mu_hist_p(3, :))], 'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
hold(ax5, 'off')
xlabel(ax5, 'Time [s]')
ylabel(ax5, 'Ballistic coefficient [m^2/kg]')
legend(ax5, {'$x_3$ (true)', '$\mu_3$', '$\mu_3 \pm 3\sigma_3$'}, 'interpreter', 'latex')
grid(ax5, 'on')

ax2 = subplot(3, 2, 2, 'Parent', hf);
semilogy(ax2, t_hist, sigma_hist(1, :))
title(ax2, 'Marginal standard deviations $\sigma_i = \sqrt{P_{ii}}$', 'interpreter', 'latex', 'FontSize', 16)
xlabel(ax2, 'Time [s]')
ylabel(ax2, 'Height [m]')
legend(ax2, '$\sigma_1$', 'interpreter', 'latex')
grid(ax2, 'on')

ax4 = subplot(3, 2, 4, 'Parent', hf);
semilogy(ax4, t_hist, sigma_hist(2, :))
xlabel(ax4, 'Time [s]')
ylabel(ax4, 'Velocity [m/s]')
legend(ax4, '$\sigma_2$', 'interpreter', 'latex')
grid(ax4, 'on')

ax6 = subplot(3, 2, 6, 'Parent', hf);
semilogy(ax6, t_hist, sigma_hist(3, :))
xlabel(ax6, 'Time [s]')
ylabel(ax6, 'Ballistic coefficient [m^2/kg]')
legend(ax6, '$\sigma_3$', 'interpreter', 'latex')
grid(ax6, 'on')

% Share common time axis zoom
ax1.XLimitMethod = 'tight';
linkprop([ax1, ax2, ax3, ax4, ax5, ax6], {'XLim'});
