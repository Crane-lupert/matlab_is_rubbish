%% Q3_VaR_Bond.m
% VAR OF A BOND
% This script calculates the yield to maturity (YTM) for a 10-year bond with annual coupon payments,
% and then computes the probability of a 10% decline in price over 30 days.
% It estimates VaR at a 99% confidence level across various horizons using:
%   1. Exact formula
%   2. Delta approximation (first order)
%   3. Delta-Gamma approximation (second order)
%   4. Monte Carlo simulation with delta approximation
%   5. Monte Carlo simulation with delta-gamma approximation
%   6. Monte Carlo simulation with full revaluation
% It also computes the Expected Shortfall (ES) using full revaluation.
%
% Assumptions:
% - Face value = 100, Coupon rate = 5%, Maturity = 10 years, Current Price = 99
% - A year = 360 days, daily fluctuations in YTM ~ N(0, 0.006)
% - At least 10,000 Monte Carlo simulations are used.

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

% Confidence level for VaR and ES
conf_level = 0.99;
% For VaR calculations, we use the absolute quantile (e.g., norminv(0.01) is ~ -2.33)
z = abs(norminv(1-conf_level,0,1));  % z ~ 2.33

%% Calculate Yield to Maturity (YTM)
% Solve: annual_coupon*sum_{t=1}^{maturity}(1/(1+x)^t) + FV/(1+x)^maturity = P0
syms x positive
eqn = annual_coupon*sum(1./(1+x).^(1:maturity)) + FV/(1+x)^maturity == P0;
sol = vpasolve(eqn, x, [0,1]);
YTM = double(sol);
fprintf('Yield to Maturity (annual): %.4f\n', YTM);

%% Compute Duration and Convexity (Analytical Approximations)
t = (1:maturity)';
% Bond price using YTM
price_calc = annual_coupon * sum(1./(1+YTM).^(t)) + FV/(1+YTM)^maturity;
% Macaulay Duration
duration = (annual_coupon*sum(t./(1+YTM).^(t)) + maturity*FV/(1+YTM)^maturity) / price_calc;
% Approximate Convexity:
convexity = (annual_coupon*sum(t.*(t+1)./(1+YTM).^(t+2)) + maturity*(maturity+1)*FV/(1+YTM)^(maturity+2)) / P0;

%% Estimate probability of a 10% decline in bond price within 30 days
% Linear approximation: dP/dy = -P0*duration/(1+YTM)
% A 10% decline means new price <= 0.9*P0, so approximate yield change required:
dy_required = -((P0 - 0.9*P0) * (1+YTM))/(P0*duration);  % negative value indicates increase in yield
% Over 30 days, yield change ~ N(0, daily_std*sqrt(30))
prob_decline = normcdf(dy_required, daily_mean, daily_std*sqrt(30));
fprintf('Probability of 10%% decline in 30 days: %.4f\n', prob_decline);

%% VaR Calculations using Exact, Delta, Delta-Gamma formulas
num_horizons = length(horizons);
VaR_exact = zeros(num_horizons,1);
VaR_delta = zeros(num_horizons,1);
VaR_delta_gamma = zeros(num_horizons,1);

% For exact formula, we assume price loss scales linearly:
for i = 1:num_horizons
    horizon = horizons(i);
    % Exact formula: Loss = P0 * z * daily_std * sqrt(horizon)
    VaR_exact(i) = P0 * z * daily_std * sqrt(horizon);
    % Delta approximation: loss ≈ -(dP/dy)*dy with dP/dy = -P0*duration/(1+YTM)
    VaR_delta(i) = (P0*duration/(1+YTM)) * z * daily_std * sqrt(horizon);
    % Delta-Gamma approximation:
    % Loss ≈ (P0*duration/(1+YTM))*dy + 0.5*P0*convexity*dy^2, with dy = -z*daily_std*sqrt(horizon)
    dy = z * daily_std * sqrt(horizon);
    VaR_delta_gamma(i) = (P0*duration/(1+YTM))*dy + 0.5 * P0 * convexity * dy^2;
end

%% Monte Carlo Simulations (10,000 simulations)
num_sim = 10000;
VaR_delta_MC = zeros(num_horizons,1);
VaR_delta_gamma_MC = zeros(num_horizons,1);
VaR_full_MC = zeros(num_horizons,1);

% Pre-calculate sensitivity parameters
dPdy = -P0 * duration/(1+YTM);   % Delta (negative)
d2Pdy2 = P0 * convexity;          % Gamma

for i = 1:num_horizons
    horizon = horizons(i);
    % Simulate yield changes over the horizon:
    dy_sim = normrnd(daily_mean, daily_std*sqrt(horizon), num_sim, 1);
    
    % Delta approximation: approximate new price P1
    P1 = P0 + dPdy * dy_sim;
    loss_delta = P0 - P1;  % loss (positive if price drops)
    VaR_delta_MC(i) = quantile(loss_delta, conf_level);
    
    % Delta-Gamma approximation:
    P2 = P0 + dPdy * dy_sim + 0.5 * d2Pdy2 * dy_sim.^2;
    loss_delta_gamma = P0 - P2;
    VaR_delta_gamma_MC(i) = quantile(loss_delta_gamma, conf_level);
    
    % Full revaluation:
    newPrices = zeros(num_sim,1);
    for j = 1:num_sim
        newYTM = YTM + dy_sim(j);
        newPrices(j) = annual_coupon * sum(1./(1+newYTM).^(1:maturity)) + FV/(1+newYTM)^maturity;
    end
    losses_full = P0 - newPrices;
    VaR_full_MC(i) = quantile(losses_full, conf_level);
end

%% Expected Shortfall Calculation (using full revaluation)
ES_full = zeros(num_horizons,1);
for i = 1:num_horizons
    horizon = horizons(i);
    dy_sim = normrnd(daily_mean, daily_std*sqrt(horizon), num_sim, 1);
    newPrices = zeros(num_sim,1);
    for j = 1:num_sim
        newYTM = YTM + dy_sim(j);
        newPrices(j) = annual_coupon * sum(1./(1+newYTM).^(1:maturity)) + FV/(1+newYTM)^maturity;
    end
    losses = P0 - newPrices;  % loss = P0 - Price, positive if price drops
    VaR_full = quantile(losses, conf_level);  
    ES_full(i) = mean(losses(losses >= VaR_full));
end

%% Display Results
VarTable = table(horizons', VaR_exact, VaR_delta, VaR_delta_gamma, VaR_delta_MC, VaR_delta_gamma_MC, VaR_full_MC, ...
    'VariableNames', {'Days', 'ExactFormula', 'DeltaApprox', 'DeltaGammaApprox', 'MC_Delta', 'MC_DeltaGamma', 'MC_Full'});
disp('VaR Estimates:');
disp(VarTable);

ESTable = table(horizons', ES_full, 'VariableNames', {'Days', 'ExpectedShortfall'});
disp('Expected Shortfall (Full Revaluation MC):');
disp(ESTable);
