%% Q3_VaR_Bond.m
% VAR OF A BOND
% This script calculates the yield to maturity (YTM) for a bond with 10-year maturity
% and annual coupon payments, and then computes the probability of a 10% decline in price
% over 30 days. It estimates VaR at a 99% confidence level across various horizons using:
% 1. Exact formula
% 2. Delta approximation (first order)
% 3. Delta-Gamma approximation (second order)
% 4. Monte Carlo simulation with delta approximation
% 5. Monte Carlo simulation with delta-gamma approximation
% 6. Monte Carlo simulation with full revaluation
% It also computes Expected Shortfall (ES) for different horizons.
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
z_score = norminv(1-conf_level,0,1);  % For 99% VaR

%% Calculate Yield to Maturity (YTM)
% Solve: annual_coupon*sum_{t=1}^{maturity}(1/(1+x)^t) + FV/(1+x)^maturity = P0
syms x positive
eqn = annual_coupon*sum(1./(1+x).^(1:maturity)) + FV/(1+x)^maturity == P0;
sol = vpasolve(eqn, x, [0,1]);
YTM = double(sol);
fprintf('Yield to Maturity (annual): %.4f\n', YTM);

%% Estimate probability of 10% decline in bond price within 30 days
% We assume daily yield fluctuations ~ N(0, daily_std) over 30 days.
% For a 10% decline, the bond price should fall to <= 0.9*P0.
% Here, using a simplified linear approximation: 
% P_change ~ P0 * duration * dy/(1+YTM)
% First, compute bond price duration.
t = 1:maturity;
% Price of bond:
price_calc = annual_coupon*sum(1./(1+YTM).^(t)) + FV/(1+YTM)^maturity;
duration = (annual_coupon*sum(t./(1+YTM).^(t)) + maturity*FV/(1+YTM)^maturity) / price_calc;
% For a 10% decline, required yield change (dy) is:
dy_required = ((0.9*P0 - P0)* (1+YTM))/(P0*duration);  % approximate linear relation
% Probability that yield change exceeds dy_required in 30 days:
prob_decline = 1 - normcdf(dy_required, daily_mean, daily_std*sqrt(30));
fprintf('Probability of 10%% decline in 30 days: %.4f\n', prob_decline);

%% VaR Calculations using Exact, Delta, Delta-Gamma formulas
% For simplicity, we use the following approximations:
% 1. Exact Formula (linear scaling): VaR_exact = |z_score| * daily_std * sqrt(horizon) * P0.
% 2. Delta Approximation: Using duration.
% 3. Delta-Gamma Approximation: Using duration and an approximate convexity.
% We simulate yield change dy ~ N(0, daily_std*sqrt(horizon)) for delta approximations.

num_horizons = length(horizons);
VaR_exact = zeros(num_horizons,1);
VaR_delta = zeros(num_horizons,1);
VaR_delta_gamma = zeros(num_horizons,1);

% Approximate convexity: second derivative of price w.r.t yield.
% A rough approximation for convexity can be computed as:
convexity = (annual_coupon*sum(t.*(t+1)./(1+YTM).^(t+2)) + maturity*(maturity+1)*FV/(1+YTM)^(maturity+2)) / P0;

for i = 1:num_horizons
    horizon = horizons(i);
    % Exact formula (as provided in sample code)
    VaR_exact(i) = z_score * daily_std * sqrt(horizon) * P0;
    % Delta approximation: approximate price change = -P0*duration*dy/(1+YTM)
    % Using dy = z_score * daily_std * sqrt(horizon)
    dy = z_score * daily_std * sqrt(horizon);
    P_delta = P0 + (P0 * duration * dy)/(1+YTM);
    VaR_delta(i) = z_score * daily_std * sqrt(horizon) * P_delta;
    % Delta-Gamma approximation:
    P_delta_gamma = P0 + (P0 * duration * dy)/(1+YTM) + 0.5 * P0 * convexity * dy^2;
    VaR_delta_gamma(i) = z_score * daily_std * sqrt(horizon) * P_delta_gamma;
end

%% Monte Carlo Simulations (10,000 simulations)
num_sim = 10000;
VaR_delta_MC = zeros(num_horizons,1);
VaR_delta_gamma_MC = zeros(num_horizons,1);
VaR_full_MC = zeros(num_horizons,1);

% Pre-calculate Black-Scholes-like full revaluation for bond.
% For bonds, full revaluation means computing the new price exactly using the bond pricing formula.
for i = 1:num_horizons
    horizon = horizons(i);
    % Simulate yield changes over horizon
    dy_sim = normrnd(0, daily_std*sqrt(horizon), num_sim, 1);
    % For delta approximation:
    P1 = P0 + (P0 * duration * dy_sim)/(1+YTM);
    VaR_delta_MC(i) = -mean(P0 - P1); % average loss
    % For delta-gamma approximation:
    P2 = P0 + (P0 * duration * dy_sim)/(1+YTM) + 0.5 * P0 * convexity * dy_sim.^2;
    VaR_delta_gamma_MC(i) = -mean(P0 - P2);
    % Full revaluation:
    % New bond price calculated exactly:
    % Price = annual_coupon * sum(1/(1+(YTM+dy))^t) + FV/(1+(YTM+dy))^maturity
    newPrices = zeros(num_sim,1);
    for j = 1:num_sim
        newYTM = YTM + dy_sim(j);
        newPrices(j) = annual_coupon*sum(1./(1+newYTM).^(1:maturity)) + FV/(1+newYTM)^maturity;
    end
    VaR_full_MC(i) = -mean(P0 - newPrices);
end

%% Expected Shortfall Calculation (using full revaluation)
ES_full = zeros(num_horizons,1);
for i = 1:num_horizons
    horizon = horizons(i);
    dy_sim = normrnd(0, daily_std*sqrt(horizon), num_sim, 1);
    newPrices = zeros(num_sim,1);
    for j = 1:num_sim
        newYTM = YTM + dy_sim(j);
        newPrices(j) = annual_coupon*sum(1./(1+newYTM).^(1:maturity)) + FV/(1+newYTM)^maturity;
    end
    % Compute losses: L = P0 - newPrice
    losses = P0 - newPrices;
    VaR_full = quantile(losses, 0.99);  % 99% VaR from simulation
    ES_full(i) = mean(losses(losses >= VaR_full));
end

%% Display Results
VarTable = table(horizons', VaR_exact, VaR_delta, VaR_delta_gamma, VaR_delta_MC, VaR_delta_gamma_MC, VaR_full_MC, 'VariableNames', ...
    {'Days', 'ExactFormula', 'DeltaApprox', 'DeltaGammaApprox', 'MC_Delta', 'MC_DeltaGamma', 'MC_Full'});
disp('VaR Estimates:');
disp(VarTable);

ESTable = table(horizons', ES_full, 'VariableNames', {'Days', 'ExpectedShortfall'});
disp('Expected Shortfall (Full Revaluation MC):');
disp(ESTable);
