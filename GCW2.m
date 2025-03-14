%% Q2_RiskParityPortfolio.m
% THE RISK PARITY PORTFOLIO
%
% This script re-uses the dataset from Q1 (stored in 'big_6_adj_closing_prices.xlsx')
% to construct four portfolios:
%   1. Equally Weighted Portfolio
%   2. Minimum Variance Portfolio
%   3. Risk Parity Portfolio (each asset contributes equally to the Component VaR)
%   4. Maximum Diversification Portfolio
%
% The sample is split into two parts:
%   - The first half of the sample is used to determine portfolio compositions (in-sample).
%   - The second half is used to compute daily log returns for each portfolio (out-of-sample).
%
% The script then evaluates performance using:
%   - Sharpe Ratio (assuming zero risk-free rate)
%   - Maximum Drawdown
%   - Number of VaR Violations at a 95% confidence level
%
% It also computes component VaR using a parametric approach (sample covariance matrix)
% and a non-parametric approach.
%
% All code and comments are in English.

%% 1. Data Preparation and Portfolio Returns Calculation
clear; close all; clc;
format short;

% Load data from the Excel file
filename = 'big_6_adj_closing_prices.xlsx';
dataTable = readtable(filename);

% Rename columns for easier access.
% Expected columns: Month, Day, Year, Alphabet, Amazon, Apple, IBM, Microsoft, Nvidia, dollar
dataTable.Properties.VariableNames = {'Month','Day','Year','Alphabet','Amazon','Apple','IBM','Microsoft','Nvidia','dollar'};

% Construct date strings using numeric month, day, and year in "M-d-yyyy" format.
datesStr = strcat(string(dataTable.Month), '-', string(dataTable.Day), '-', string(dataTable.Year));
datesStr = regexprep(datesStr, '\s+', ' ');
datesStr = strtrim(datesStr);
dates_all = datetime(datesStr, 'InputFormat', 'M-d-yyyy');

% Restrict the data to the period from 1-1-2014 to 12-31-2024 (consistent with Q1)
start_period = datetime('1-1-2014','InputFormat','M-d-yyyy');
end_period   = datetime('12-31-2024','InputFormat','M-d-yyyy');
maskPeriod = (dates_all >= start_period) & (dates_all <= end_period);
dates_all = dates_all(maskPeriod);
dataTable = dataTable(maskPeriod, :);

% Sort data by date in ascending order
[dates_all, sortIdx] = sort(dates_all);
dataTable = dataTable(sortIdx, :);

% Extract adjusted closing prices for the six stocks:
% Stocks: Alphabet, Amazon, Apple, IBM, Microsoft, Nvidia.
tickers = {'Alphabet','Amazon','Apple','IBM','Microsoft','Nvidia'};
prices = table2array(dataTable(:, tickers));

% Compute daily log returns (vectorized)
log_returns = diff(log(prices));
dates_returns = dates_all(2:end);

% Build equally weighted portfolio returns (for reference)
num_assets = numel(tickers);
w_eq = ones(num_assets,1)/num_assets;
portfolio_returns = log_returns * w_eq;

fprintf('Data preparation and portfolio returns calculation complete.\n');

%% 2. Split the Data into In-Sample and Out-of-Sample
% First half (in-sample) is used for portfolio composition,
% the second half (out-of-sample) is used for performance evaluation.
num_obs = size(log_returns,1);
split_idx = floor(num_obs/2);
ret_in = log_returns(1:split_idx, :);
ret_out = log_returns(split_idx+1:end, :);

%% 3. Portfolio Composition: Compute Component VaR (In-Sample)
% Compute the sample covariance matrix using in-sample returns.
Sigma = cov(ret_in);

% Set confidence level for VaR calculations (95%)
alpha = 0.95;
z = norminv(1-alpha, 0, 1);  % Note: norminv(0.05) is negative

% Equally weighted portfolio:
w_eq = ones(num_assets,1)/num_assets;
sg2_eq = w_eq' * Sigma * w_eq;
VaR_param_eq = -z * sqrt(sg2_eq);  % Parametric VaR for full portfolio
MVaR_eq = -z * (Sigma * w_eq) / sqrt(sg2_eq);  % Marginal VaR for each asset
CVaR_eq = w_eq .* MVaR_eq;                    % Component VaR = weight * marginal VaR
CVaR_eq_pct = CVaR_eq / sum(CVaR_eq);           % Percentage contributions

T_Component_eq = table(w_eq, MVaR_eq, CVaR_eq, CVaR_eq_pct, 'VariableNames', {'Weights','MVaR','CVaR','CVaR_Percent'});
T_Component_eq.Properties.RowNames = tickers;
disp('Component VaR for Equally Weighted Portfolio (Parametric):');
disp(T_Component_eq);

% Non-parametric VaR for equally weighted portfolio (in-sample)
VaR_np_eq = -prctile(portfolio_returns(1:split_idx), (1-alpha)*100);
fprintf('Equally weighted portfolio parametric VaR: %.4f, non-parametric VaR: %.4f\n', VaR_param_eq, VaR_np_eq);

%% 4. Portfolio Optimization
% Define common starting point and options for optimization.
x0 = ones(num_assets,1)/num_assets;
options = optimoptions('fmincon','Display','off');

% 4a. Minimum Variance Portfolio
w_mv = fmincon(@(x) x'*Sigma*x, x0, [], [], ones(1,num_assets), 1, zeros(num_assets,1), ones(num_assets,1), [], options);
sg2_mv = w_mv' * Sigma * w_mv;
MVaR_mv = -z * (Sigma * w_mv) / sqrt(sg2_mv);
CVaR_mv = w_mv .* MVaR_mv;
CVaR_mv_pct = CVaR_mv / sum(CVaR_mv);

% 4b. Risk Parity Portfolio
% Objective: minimize the sum of squared deviations of individual risk contributions
risk_parity_obj = @(x) sum((x .* (Sigma*x) - (x'*Sigma*x)/num_assets).^2);
w_rp = fmincon(risk_parity_obj, x0, [], [], ones(1,num_assets), 1, zeros(num_assets,1), ones(num_assets,1), [], options);
sg2_rp = w_rp' * Sigma * w_rp;
MVaR_rp = -z * (Sigma * w_rp) / sqrt(sg2_rp);
CVaR_rp = w_rp .* MVaR_rp;
CVaR_rp_pct = CVaR_rp / sum(CVaR_rp);

% 4c. Maximum Diversification Portfolio
sigma_vec = std(ret_in)';  % asset volatilities (sample standard deviations)
obj_md = @(x) - ((x' * sigma_vec) / sqrt(x'*Sigma*x));
w_md = fmincon(obj_md, x0, [], [], ones(1,num_assets), 1, zeros(num_assets,1), ones(num_assets,1), [], options);
sg2_md = w_md' * Sigma * w_md;
MVaR_md = -z * (Sigma * w_md) / sqrt(sg2_md);
CVaR_md = w_md .* MVaR_md;
CVaR_md_pct = CVaR_md / sum(CVaR_md);

% Create a table for portfolio weights
T_Portfolios = table(w_eq, w_mv, w_rp, w_md, 'VariableNames', {'EquallyWeighted','MinVariance','RiskParity','MaxDiversification'});
T_Portfolios.Properties.RowNames = tickers;
disp('Portfolio Weights:');
disp(T_Portfolios);

%% 5. Performance Evaluation on Out-of-Sample Data
% Compute out-of-sample returns for each portfolio
ret_out_eq = ret_out * w_eq;
ret_out_mv = ret_out * w_mv;
ret_out_rp = ret_out * w_rp;
ret_out_md = ret_out * w_md;

% Compute portfolio value paths (assuming starting value 1, using log returns)
value_eq = exp(cumsum(ret_out_eq));
value_mv = exp(cumsum(ret_out_mv));
value_rp = exp(cumsum(ret_out_rp));
value_md = exp(cumsum(ret_out_md));

% Compute Sharpe ratios (assuming zero risk-free rate)
Sharpe_eq = mean(ret_out_eq) / std(ret_out_eq);
Sharpe_mv = mean(ret_out_mv) / std(ret_out_mv);
Sharpe_rp = mean(ret_out_rp) / std(ret_out_rp);
Sharpe_md = mean(ret_out_md) / std(ret_out_md);

% Define maximum drawdown function
maxDD = @(v) max((cummax(v)-v)./cummax(v));
MaxDD_eq = maxDD(value_eq);
MaxDD_mv = maxDD(value_mv);
MaxDD_rp = maxDD(value_rp);
MaxDD_md = maxDD(value_md);

% Compute out-of-sample VaR (using standard deviation method) at 95% confidence.
conf_VaR = 0.95;
z_VaR = norminv(1-conf_VaR,0,1);
VaR_eq_out = -z_VaR * std(ret_out_eq);
VaR_mv_out = -z_VaR * std(ret_out_mv);
VaR_rp_out = -z_VaR * std(ret_out_rp);
VaR_md_out = -z_VaR * std(ret_out_md);

% Count VaR violations (number of days with loss below VaR threshold)
violations_eq = sum(ret_out_eq <= -VaR_eq_out);
violations_mv = sum(ret_out_mv <= -VaR_mv_out);
violations_rp = sum(ret_out_rp <= -VaR_rp_out);
violations_md = sum(ret_out_md <= -VaR_md_out);

% Create a table of performance measures
Measures_Performance = table([Sharpe_eq; Sharpe_mv; Sharpe_rp; Sharpe_md],...
    [MaxDD_eq; MaxDD_mv; MaxDD_rp; MaxDD_md],...
    [violations_eq; violations_mv; violations_rp; violations_md],...
    'VariableNames', {'SharpeRatio','MaxDrawdown','VaRViolations'},...
    'RowNames', {'EquallyWeighted','MinVariance','RiskParity','MaxDiversification'});
disp('Out-of-Sample Performance Measures:');
disp(Measures_Performance);

%% 6. Plot Component VaR Pie Charts (In-Sample)
figure('Color',[1 1 1])
colors = jet(num_assets);
subplot(1,3,1);
pie(CVaR_eq_pct, tickers);
colormap(colors);
title('Component VaR % - Equally Weighted','Interpreter','latex');

subplot(1,3,2);
pie(CVaR_mv_pct, tickers);
colormap(colors);
title('Component VaR % - Minimum Variance','Interpreter','latex');

subplot(1,3,3);
pie(CVaR_rp_pct, tickers);
colormap(colors);
title('Component VaR % - Risk Parity','Interpreter','latex');

%% 7. Additional VaR Calculation (Out-of-Sample Rolling Window)
% Using the out-of-sample returns of the Risk Parity portfolio for VaR estimation.
RP_port_ret2 = ret_out * w_rp;
% (For reference, equally weighted portfolio values can also be computed)
Port_value_EW = prices(split_idx+1:end,:) * w_eq;
Port_value_RP = prices(split_idx+1:end,:) * w_rp;

% Set rolling window parameters for VaR estimation (120-day window, same as Q1)
n = 1;
conf_interval = 0.95;
window_length = 120;  % Changed from 300 to 120 days
num_roll = size(RP_port_ret2,1) - window_length;

% Pre-allocate arrays for VaR values using different methods (computed over each 120-day window)
VaR_Parametric_loop = zeros(num_roll,1);
VaR_MHS_loop = zeros(num_roll,1);
VaR_non_Parametric_loop = zeros(num_roll,1);

for i = 1:num_roll
    data_window = RP_port_ret2(i:i+window_length-1);
    VaR_Parametric_loop(i) = calcVaR(data_window, conf_interval, 'Gaussian');
    VaR_MHS_loop(i) = calcVaR(data_window, conf_interval, 'MHS');
    VaR_non_Parametric_loop(i) = -prctile(data_window, (1-conf_interval)*100);
end

%% Plot: VaRs Over Time at 95% Confidence Level for Risk Parity Portfolio
ZoomInd = 1:num_roll;
a_plot = VaR_Parametric_loop;
b_plot = VaR_MHS_loop;
c_plot = VaR_non_Parametric_loop;

figure('Color',[1 1 1])
bar(ZoomInd, RP_port_ret2(ZoomInd), 0.5, 'FaceColor',[0.7 0.7 0.7]);
hold on;
plot(-a_plot, 'b-', 'LineWidth', 1.5);
plot(-b_plot, 'r--', 'LineWidth', 1.5);
plot(-c_plot, 'g:', 'LineWidth', 1.5);
xlabel('Time (Days)','Interpreter','latex')
ylabel('VaR / Portfolio Returns','Interpreter','latex')
legend('Risk Parity Portfolio','Parametric VaR','Modified HS VaR','Non-Parametric VaR','Interpreter','latex')
title('VaRs Over Time at 95% Confidence','Interpreter','latex')
grid on;
hold off;

%% Plot: VaR vs Confidence Level Comparison (for the first iteration of out-of-sample)
confidences = linspace(0.01, 0.99, 100);
n_iter = num_roll;
VaR_Parametric_mat = zeros(n_iter, length(confidences));
VaR_MHS_mat = zeros(n_iter, length(confidences));
VaR_NP_mat = zeros(n_iter, length(confidences));

for j = 1:length(confidences)
    conf_val = confidences(j);
    for i = 1:n_iter
        window_data = RP_port_ret2(i:i+window_length-1);
        VaR_Parametric_mat(i,j) = calcVaR(window_data, conf_val, 'Gaussian');
        VaR_MHS_mat(i,j) = calcVaR(window_data, conf_val, 'MHS');
        VaR_NP_mat(i,j) = calcVaR(window_data, conf_val, 'HS');
    end
end

figure('Color',[1 1 1])
subplot(2,2,1)
plot(confidences, VaR_Parametric_mat(1,:), 'LineWidth', 1.5);
title('Parametric VaR vs Confidence','Interpreter','latex')
xlabel('Confidence Level','Interpreter','latex')
ylabel('VaR','Interpreter','latex')
grid on

subplot(2,2,2)
plot(confidences, VaR_NP_mat(1,:), 'LineWidth', 1.5);
title('Non-Parametric VaR vs Confidence','Interpreter','latex')
xlabel('Confidence Level','Interpreter','latex')
ylabel('VaR','Interpreter','latex')
grid on

subplot(2,2,3)
plot(confidences, VaR_MHS_mat(1,:), 'LineWidth', 1.5);
title('Modified HS VaR vs Confidence','Interpreter','latex')
xlabel('Confidence Level','Interpreter','latex')
ylabel('VaR','Interpreter','latex')
grid on

subplot(2,2,4)
% For demonstration, we plot the parametric VaR as a placeholder for Monte Carlo VaR.
plot(confidences, VaR_Parametric_mat(1,:), 'LineWidth', 1.5);
title('Monte Carlo VaR vs Confidence (Placeholder)','Interpreter','latex')
xlabel('Confidence Level','Interpreter','latex')
ylabel('VaR','Interpreter','latex')
grid on;


%% End of Script

%% Function: calcVaR
function VaR = calcVaR(returnData, confLevel, method)
% calcVaR calculates a single VaR value for a given vector of returns,
% a given confidence level, and the specified method.
% Inputs:
%   returnData - Vector of return data.
%   confLevel  - Confidence level (e.g., 0.90 or 0.99).
%   method     - VaR calculation method, one of 'HS', 'Gaussian', 'MHS', or 'DeltaNorm'.
% Output:
%   VaR        - The calculated VaR value (a scalar).
%
% For Modified HS, an EWMA-based time weighting is used.
alpha = 1 - confLevel;  % Tail probability
switch method
    case 'HS'
        VaR = -prctile(returnData, alpha*100);
    case 'Gaussian'
        mu = mean(returnData);
        sigma = std(returnData);
        z = norminv(alpha);
        VaR = -(mu + sigma*z);
    case 'MHS'
        % Modified HS using EWMA weighting
        T = length(returnData);
        lambda = 0.94; % Decay factor
        weights = lambda.^( (T-1):-1:0 )';
        weights = weights / sum(weights);
        [sortedData, sortIdx] = sort(returnData);
        sortedWeights = weights(sortIdx);
        cumWeights = cumsum(sortedWeights);
        idx = find(cumWeights >= alpha, 1, 'first');
        VaR = -sortedData(idx);
    case 'DeltaNorm'
        mu = mean(returnData);
        sigma = std(returnData);
        z = norminv(alpha);
        VaR = -(mu + sigma*z);
    otherwise
        VaR = NaN;
end
end
