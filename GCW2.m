%% Q2_RiskParityPortfolio.m
% THE RISK PARITY PORTFOLIO
%
% This script re-uses the dataset from Q1 (stored in 'big_6_adj_closing_prices.xlsx')
% to construct three portfolios:
%   1. Equally Weighted Portfolio
%   2. Minimum Variance Portfolio
%   3. Risk Parity Portfolio (each asset contributes equally to the Component VaR)
%
% The sample is split into two parts:
%   - The first half of the sample is used to determine portfolio compositions.
%   - The second half is used to compute daily log returns for each portfolio.
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

% Since the Month column is numeric (e.g., 9 for September), construct date strings directly.
% Use the "M-d-yyyy" format consistently.
datesStr = strcat(string(dataTable.Month), '-', string(dataTable.Day), '-', string(dataTable.Year));
datesStr = regexprep(datesStr, '\s+', ' ');
datesStr = strtrim(datesStr);
dates_all = datetime(datesStr, 'InputFormat', 'M-d-yyyy');

% Restrict the data to the period from 1-1-2014 to 12-31-2023 (using M-d-yyyy format)
start_period = datetime('1-1-2014','InputFormat','M-d-yyyy');
end_period   = datetime('12-31-2023','InputFormat','M-d-yyyy');
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
% Use the first half of the observations for portfolio composition (in-sample)
% and the second half for performance evaluation (out-of-sample).
num_obs = size(log_returns,1);
split_idx = floor(num_obs/2);
ret_in = log_returns(1:split_idx, :);
ret_out = log_returns(split_idx+1:end, :);

%% 3. Portfolio Composition: Compute Component VaR (In-Sample)
% Determine portfolio weights using the in-sample data.
Sigma = cov(ret_in);

% Set confidence level for VaR calculations (95%)
alpha = 0.95;
z = norminv(1-alpha, 0, 1);

% Equally weighted portfolio:
w_eq = ones(num_assets,1)/num_assets;
sg2_eq = w_eq' * Sigma * w_eq;
VaR_param_eq = -z * sqrt(sg2_eq);
MVaR_eq = -z * (Sigma * w_eq) / sqrt(sg2_eq);
CVaR_eq = w_eq .* MVaR_eq;
CVaR_eq_pct = CVaR_eq / sum(CVaR_eq);

T_Component_eq = table(w_eq, MVaR_eq, CVaR_eq, CVaR_eq_pct, 'VariableNames', {'Weights','MVaR','CVaR','CVaR_Percent'});
T_Component_eq.Properties.RowNames = tickers;
disp('Component VaR for Equally Weighted Portfolio (Parametric):');
disp(T_Component_eq);

VaR_np_eq = -prctile(portfolio_returns(1:split_idx), (1-alpha)*100);
fprintf('Equally weighted portfolio parametric VaR: %.4f, non-parametric VaR: %.4f\n', VaR_param_eq, VaR_np_eq);

%% 4. Portfolio Optimization
% 4a. Minimum Variance Portfolio
x0 = ones(num_assets,1)/num_assets;
options = optimoptions('fmincon','Display','off');
w_mv = fmincon(@(x) x'*Sigma*x, x0, [], [], ones(1,num_assets), 1, zeros(num_assets,1), ones(num_assets,1), [], options);
sg2_mv = w_mv' * Sigma * w_mv;
MVaR_mv = -z * (Sigma * w_mv) / sqrt(sg2_mv);
CVaR_mv = w_mv .* MVaR_mv;
CVaR_mv_pct = CVaR_mv / sum(CVaR_mv);

% 4b. Risk Parity Portfolio
x0 = ones(num_assets,1)/num_assets;
w_rp = fmincon(@(x) std(x .* (Sigma*x)/sqrt(x'*Sigma*x)), x0, [], [], ones(1,num_assets), 1, zeros(num_assets,1), ones(num_assets,1), [], options);
sg2_rp = w_rp' * Sigma * w_rp;
MVaR_rp = -z * (Sigma * w_rp) / sqrt(sg2_rp);
CVaR_rp = w_rp .* MVaR_rp;
CVaR_rp_pct = CVaR_rp / sum(CVaR_rp);

% 4c. Maximum Diversification Portfolio
sigma_vec = std(ret_in)';
obj_md = @(x) - ((x' * sigma_vec) / sqrt(x'*Sigma*x));
w_md = fmincon(obj_md, x0, [], [], ones(1,num_assets), 1, zeros(num_assets,1), ones(num_assets,1), [], options);
sg2_md = w_md' * Sigma * w_md;
MVaR_md = -z * (Sigma * w_md) / sqrt(sg2_md);
CVaR_md = w_md .* MVaR_md;
CVaR_md_pct = CVaR_md / sum(CVaR_md);

T_Portfolios = table(w_eq, w_mv, w_rp, w_md, 'VariableNames', {'EquallyWeighted','MinVariance','RiskParity','MaxDiversification'});
T_Portfolios.Properties.RowNames = tickers;
disp('Portfolio Weights:');
disp(T_Portfolios);

%% 5. Performance Evaluation on Out-of-Sample Data
ret_out_eq = ret_out * w_eq;
ret_out_mv = ret_out * w_mv;
ret_out_rp = ret_out * w_rp;
ret_out_md = ret_out * w_md;

value_eq = exp(cumsum(ret_out_eq));
value_mv = exp(cumsum(ret_out_mv));
value_rp = exp(cumsum(ret_out_rp));
value_md = exp(cumsum(ret_out_md));

Sharpe_eq = mean(ret_out_eq) / std(ret_out_eq);
Sharpe_mv = mean(ret_out_mv) / std(ret_out_mv);
Sharpe_rp = mean(ret_out_rp) / std(ret_out_rp);
Sharpe_md = mean(ret_out_md) / std(ret_out_md);

maxDD = @(v) max((cummax(v)-v)./cummax(v));

MaxDD_eq = maxDD(value_eq);
MaxDD_mv = maxDD(value_mv);
MaxDD_rp = maxDD(value_rp);
MaxDD_md = maxDD(value_md);

conf_VaR = 0.95;
z_VaR = norminv(1-conf_VaR,0,1);
VaR_eq_out = -z_VaR * std(ret_out_eq);
VaR_mv_out = -z_VaR * std(ret_out_mv);
VaR_rp_out = -z_VaR * std(ret_out_rp);
VaR_md_out = -z_VaR * std(ret_out_md);

violations_eq = sum(ret_out_eq <= -VaR_eq_out);
violations_mv = sum(ret_out_mv <= -VaR_mv_out);
violations_rp = sum(ret_out_rp <= -VaR_rp_out);
violations_md = sum(ret_out_md <= -VaR_md_out);

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
% Using the second half (ret_out) for the Risk Parity portfolio.
RP_port_ret2 = ret_out * w_rp;
EW_port_ret2 = ret_out * w_eq;
Port_value_EW = prices(split_idx+1:end,:) * w_eq;
Port_value_RP = prices(split_idx+1:end,:) * w_rp;

% Rolling window parameters for VaR estimation
n = 1;
conf_interval = 0.95;
window_length = 300;
VaR_Parametric_loop = zeros(size(RP_port_ret2,1)-window_length,1);
VaR_MHS_loop = zeros(size(RP_port_ret2,1)-window_length,1);
for i = 1:size(RP_port_ret2,1)-window_length
    data_window = RP_port_ret2(i:i+window_length-1);
    VaR_Parametric_loop(i) = calcVaR(data_window, conf_interval, 'Gaussian');
    VaR_MHS_loop(i) = calcVaR(data_window, conf_interval, 'MHS');
end
VaR_non_Parametric = -prctile(RP_port_ret2, (1-conf_interval)*100);

%% Plot: VaRs Over Time at 95% Confidence Level for Risk Parity Portfolio
start_plot = 1;
end_plot = length(VaR_Parametric_loop);
ZoomInd = start_plot:end_plot;
a_plot = VaR_Parametric_loop;
b_plot = VaR_MHS_loop;
c_plot = repmat(VaR_non_Parametric, length(VaR_Parametric_loop), 1);

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
n_iter = size(RP_port_ret2,1) - window_length + 1;
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
% calcVaR calculates a single VaR value for a given vector of returns, confidence level, and method.
% Inputs:
%   returnData - Vector of return data.
%   confLevel  - Confidence level (e.g., 0.90 or 0.99).
%   method     - VaR calculation method, one of 'HS', 'Gaussian', 'MHS', 'DeltaNorm'.
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