%% Q3_VaR_Bond.m
% VAR OF A BOND & 10% Price Decline Probability via YTM Path Simulation
%
% This script performs the following:
% 1) Calculates the yield to maturity (YTM) for a 10-year bond with an annual 5% coupon,
%    a face value of 100, and a current price of 99.
% 2) Simulates daily YTM paths over a 30-day period and estimates the probability that the 
%    bond price declines by 10% (i.e. falls below 89.1) at any day during that period.
% 3) Computes VaR across various horizons (1, 10, 20, ... 90 days) using six methods:
%    Exact formula, Delta approximation, Delta-Gamma approximation, MC Delta, MC Delta-Gamma,
%    and MC full revaluation.
%
% Note: The bond price is a fractional function of the YTM, so an increase in YTM leads
%       to a decrease in price. Therefore, for the exact formula, the YTM shock is applied using
%       the absolute value (|z|).
%
% Assumptions:
% - Face value = 100, coupon rate = 5%, maturity = 10 years, current price = 99.
% - 360 days per year.
% - Daily YTM fluctuations ~ N(0, 0.006^2) (i.i.d.)
% - Monte Carlo simulations use at least 10,000 iterations

clear; close all; clc;
format short;

%% Parameters
FV = 100;
coupon_rate = 0.05;
annual_coupon = FV * coupon_rate;
maturity = 10; % years
P0 = 99;
days_in_year = 360;
daily_std = 0.006;
daily_mean = 0;

% Horizons (in days)
horizons = [1,10,20,30,40,50,60,70,80,90];

% Confidence level for VaR and ES (99%)
conf_level = 0.99;
% z-score for 1% quantile (for downward movement; originally negative but absolute value is used)
z_0_01 = -2.326;

%% Calculate Yield to Maturity (YTM)
% Equation: annual_coupon * sum_{t=1}^{10} (1/(1+y)^t) + FV/(1+y)^10 = 99
syms x positive
eqn = annual_coupon * sum(1./(1+x).^(1:maturity)) + FV/(1+x)^maturity == P0;
sol = vpasolve(eqn, x, [0,1]);
YTM = double(sol);
fprintf('Yield to Maturity (annual): %.4f\n', YTM);

%% Compute Duration and Convexity (Analytical Approximations)
t = (1:maturity)';
price_calc = annual_coupon * sum(1./(1+YTM).^(t)) + FV/(1+YTM)^maturity;
duration = (annual_coupon * sum(t./(1+YTM).^(t)) + maturity*FV/(1+YTM)^maturity) / price_calc;
convexity = (annual_coupon * sum(t.*(t+1)./(1+YTM).^(t+2)) + maturity*(maturity+1)*FV/(1+YTM)^(maturity+2)) / P0;
fprintf('Duration: %.4f\n', duration);
fprintf('Convexity: %.4f\n', convexity);

% Modified Duration
mod_duration = duration/(1+YTM);

%% Define bondPrice function
% Remaining maturity after h days: maturity - h/360 (in years)
% Only include coupon payments after h/360
bondPrice = @(y, h) ( ...
    sum( annual_coupon ./ (1+y).^( max((1:maturity)-h/360,0) ) .* ((1:maturity) > h/360) ) ...
    + FV/(1+y)^(maturity - h/360) );

%% 1) 10% Price Decline Probability within 30 Days via YTM Path Simulation
% For each simulation, generate 30 daily YTM increments, compute the cumulative YTM for each day,
% then calculate the bond price for each day (with remaining maturity adjusted).
% If any of the 30 bond prices is below the threshold (89.1), count that simulation as a hit.
h_decline = 30;
num_sim_decline = 10000;
threshold = 0.9 * P0;  % 89.1

count_decline = 0;
for i = 1:num_sim_decline
    % Generate 30 daily YTM changes
    dy_daily = normrnd(daily_mean, daily_std, h_decline, 1);
    % Compute cumulative YTM for each day (a 30-element vector)
    ytm_path = YTM + cumsum(dy_daily);
    %bond price 89.1
    %bond price -> 
    hit = false;
    % For each day, compute the bond price with remaining maturity = maturity - (d/360)
    for d = 1:h_decline
        price_d = bondPrice(ytm_path(d), d);
        if price_d <= threshold
            hit = true;
            break; % Exit the inner loop if threshold is hit
        end
    end
    if hit
        count_decline = count_decline + 1;
    end
end
prob_decline_path = count_decline / num_sim_decline;
fprintf('Probability of 10%% decline within 30 days (path simulation): %.4f\n', prob_decline_path);

%% 1b) 10% Price Decline Probability via Distribution Method
% Solve for the threshold yield (y_target) such that the bond price after 30 days equals the threshold (89.1).
syms y_target positive
eqn_target = bondPrice(y_target, 30) == threshold;
sol_y = vpasolve(eqn_target, y_target, [YTM, 1]);  % search in a reasonable range (from current YTM to 1)
y_target_val = double(sol_y);
fprintf('Threshold YTM for 10%% price decline after 30 days: %.4f\n', y_target_val);

% The required cumulative YTM change is:
delta_y_required = y_target_val - YTM;
fprintf('Required cumulative YTM change over 30 days: %.4f\n', delta_y_required);

% Since the cumulative YTM change over 30 days ~ N(0, (daily_std*sqrt(30))^2),
% the probability that the cumulative change is greater than or equal to delta_y_required is:
prob_decline_dist = 1 - normcdf(delta_y_required, 0, daily_std*sqrt(30));
fprintf('Probability of 10%% decline within 30 days (distribution method): %.4f\n', prob_decline_dist);


%% 2) VaR Calculations using Exact, Delta, Delta-Gamma formulas
num_horizons = length(horizons);
VaR_exact = zeros(num_horizons,1);
VaR_delta = zeros(num_horizons,1);
VaR_delta_gamma = zeros(num_horizons,1);

% Delta using modified duration: dP/dy ≈ -P0 * Duration/(1+YTM)
dPdy = -P0 * duration/(1+YTM);
% Gamma using convexity: d²P/dy² ≈ P0 * convexity
d2Pdy2 = P0 * convexity;

for i = 1:num_horizons
    h = horizons(i);
    % Shock: YTM change corresponding to the 1% quantile (using absolute value)
    dy = abs(z_0_01) * daily_std * sqrt(h);
    
    % 1) Exact formula: Revalue the bond using increased YTM (using absolute value)
    newYTM_exact = YTM + dy;
    P_h_exact = bondPrice(newYTM_exact, h);
    VaR_exact(i) = P0 - P_h_exact;  % Loss (should be positive)
    
    % 2) Delta approximation: ΔP ≈ -dP/dy * Δy
    VaR_delta(i) = -dPdy * dy;
    
    % 3) Delta-Gamma approximation: ΔP ≈ -dP/dy * dy - 0.5 * d2P/dy^2 * dy^2
    VaR_delta_gamma(i) = -dPdy * dy - 0.5 * d2Pdy2 * dy^2;
    
    % Check ordering: Expected condition is Delta Approx > Exact > Delta-Gamma Approx.
    if ~( VaR_delta(i) > VaR_exact(i) && VaR_exact(i) > VaR_delta_gamma(i) )
        warning('At horizon %d days: Ordering condition violated: Delta = %.4f, Exact = %.4f, Delta-Gamma = %.4f', ...
            h, VaR_delta(i), VaR_exact(i), VaR_delta_gamma(i));
    end
end

%% Monte Carlo Simulations for VaR (minimum 10,000 iterations)
num_sim = 10000;
VaR_delta_MC = zeros(num_horizons,1);
VaR_delta_gamma_MC = zeros(num_horizons,1);
VaR_full_MC = zeros(num_horizons,1);

for i = 1:num_horizons
    h = horizons(i);
    % Simulate YTM change over h days (each sample is the cumulative change over h days)
    dy_sim = normrnd(daily_mean, daily_std*sqrt(h), num_sim, 1);
    
    % 4) MC: Delta approximation
    P_sim_delta = P0 + dPdy * dy_sim;
    loss_delta = P0 - P_sim_delta;
    VaR_delta_MC(i) = quantile(loss_delta, conf_level);
    
    % 5) MC: Delta-Gamma approximation
    P_sim_dg = P0 + dPdy * dy_sim + 0.5 * d2Pdy2 * dy_sim.^2;
    loss_dg = P0 - P_sim_dg;
    VaR_delta_gamma_MC(i) = quantile(loss_dg, conf_level);
    
    % 6) MC: Full revaluation - revalue the bond using the single cumulative YTM change
    newPrices_MC = zeros(num_sim,1);
    for j = 1:num_sim
        newYTM = YTM + dy_sim(j);
        newPrices_MC(j) = bondPrice(newYTM, h);
    end
    losses_full = P0 - newPrices_MC;
    VaR_full_MC(i) = quantile(losses_full, conf_level);
end

%% Expected Shortfall Calculation (using full revaluation MC)
ES_full = zeros(num_horizons,1);
for i = 1:num_horizons
    h = horizons(i);
    dy_sim = normrnd(daily_mean, daily_std*sqrt(h), num_sim, 1);
    newPrices_ES = zeros(num_sim,1);
    for j = 1:num_sim
        newYTM = YTM + dy_sim(j);
        newPrices_ES(j) = bondPrice(newYTM, h);
    end
    losses_ES = P0 - newPrices_ES;
    VaR_full_current = quantile(losses_ES, conf_level);
    ES_full(i) = mean(losses_ES(losses_ES >= VaR_full_current));
end

%% Display Results and Save to CSV
VarTable = table(horizons', VaR_exact, VaR_delta, VaR_delta_gamma, VaR_delta_MC, VaR_delta_gamma_MC, VaR_full_MC, ...
    'VariableNames', {'Days', 'ExactFormula', 'DeltaApprox', 'DeltaGammaApprox', 'MC_Delta', 'MC_DeltaGamma', 'MC_Full'});
disp('VaR Estimates:');
disp(VarTable);

ESTable = table(horizons', ES_full, 'VariableNames', {'Days', 'ExpectedShortfall'});
disp('Expected Shortfall (Full Revaluation MC):');
disp(ESTable);

writetable(VarTable, 'VaR_Estimates.csv');
writetable(ESTable, 'ExpectedShortfall.csv');

%% Visualization
% Plot VaR estimates
figure;
plot(horizons, VaR_exact, '-o', 'LineWidth', 1.5); hold on;
plot(horizons, VaR_delta, '-s', 'LineWidth', 1.5);
plot(horizons, VaR_delta_gamma, '-d', 'LineWidth', 1.5);
plot(horizons, VaR_delta_MC, '-^', 'LineWidth', 1.5);
plot(horizons, VaR_delta_gamma_MC, '-v', 'LineWidth', 1.5);
plot(horizons, VaR_full_MC, '-p', 'LineWidth', 1.5);
xlabel('Horizon (days)');
ylabel('VaR');
title('VaR Estimates vs. Horizon');
legend('Exact', 'Delta', 'Delta-Gamma', 'MC Delta', 'MC Delta-Gamma', 'MC Full', 'Location', 'NorthWest');
grid on;
saveas(gcf, 'VaR_Estimates.png');

% Plot Expected Shortfall
figure;
plot(horizons, ES_full, '-o', 'LineWidth', 1.5);
xlabel('Horizon (days)');
ylabel('Expected Shortfall');
title('Expected Shortfall (Full Revaluation MC) vs. Horizon');
grid on;
saveas(gcf, 'ExpectedShortfall.png');

%% Statistical Test
% Example: Compare MC Full revaluation VaR and MC Delta-Gamma VaR using a t-test
[h_test, p_value] = ttest(VaR_full_MC, VaR_delta_gamma_MC);
fprintf('T-test (MC Full vs. MC Delta-Gamma): h = %d, p-value = %.4f\n', h_test, p_value);