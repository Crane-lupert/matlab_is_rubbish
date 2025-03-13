%% Group Coursework - Q1: VAR Modelling (Unified and Improved Code)
% This script performs the following tasks:
% 1. Data Preparation and Portfolio Returns Calculation.
% 2. Statistical Analysis of Portfolio Returns including:
%    - Descriptive statistics, histograms, QQ plots, CDF comparison, various tests.
% 3. 6-Month Rolling Window VaR Estimation and Backtesting using four methods,
%    with Kupiec and Christoffersen's Conditional Coverage tests.
% 4. 120-Day Rolling Window VaR Estimation and comparison across methods.
% 5. New Additions:
%    A. Enlarged histogram of the tail region of log returns.
%    B. Separate autocorrelation plots for absolute and squared returns.
%    C. Conditional Coverage Test results displayed in tables.
%    D. Probability Integral Transform (PIT) analysis with KS test and histogram.
%    E. VaR violation clustering analysis via stem plot.
%
% Note: All generated figures are saved in the "image" folder.
%       No plot title is added (to avoid layout inconsistencies); captions are provided via file names.

%% Setup: Create "image" folder if it does not exist
if ~exist('image','dir')
    mkdir('image');
end

%% 1. Data Preparation and Portfolio Returns Calculation
clear; close all; clc;
format short;

% Load data from the Excel file
filename = 'big_6_adj_closing_prices.xlsx';
dataTable = readtable(filename);

% Rename columns for easier access.
% Expected columns: Month, Day, Year, Alphabet, Amazon, Apple, IBM, Microsoft, Nvidia, dollar
dataTable.Properties.VariableNames = {'Month', 'Day', 'Year', 'Alphabet', 'Amazon', 'Apple', 'IBM', 'Microsoft', 'Nvidia', 'dollar'};

% Construct date strings using numeric month, day, and year (e.g., '9-30-2014')
datesStr = strcat(string(dataTable.Month), '-', string(dataTable.Day), '-', string(dataTable.Year));
datesStr = regexprep(datesStr, '\s+', ' ');
datesStr = strtrim(datesStr);
dates_all = datetime(datesStr, 'InputFormat', 'M-d-yyyy');

% Sort data by date in ascending order
[dates_all, sortIdx] = sort(dates_all);
dataTable = dataTable(sortIdx, :);

% Extract adjusted closing prices for selected stocks:
% Stocks: Alphabet, Amazon, Apple, IBM, Microsoft, Nvidia.
tickers = {'Alphabet', 'Amazon', 'Apple', 'IBM', 'Microsoft', 'Nvidia'};
prices = table2array(dataTable(:, tickers));

% Compute daily log returns (vectorized)
log_returns = diff(log(prices));
dates_returns = dates_all(2:end);

% Compute daily simple returns: (P_t / P_{t-1} - 1)
simple_returns = prices(2:end,:) ./ prices(1:end-1,:) - 1;

% Build equally weighted portfolio returns for both log and simple returns.
num_stocks = numel(tickers);
portfolio_weights = ones(1, num_stocks) / num_stocks;
portfolio_returns = log_returns * portfolio_weights';         % Log returns portfolio
portfolio_returns_simple = simple_returns * portfolio_weights'; % Simple returns portfolio

fprintf('Portfolio returns calculation complete.\n');

%% 2. Statistical Analysis of Portfolio Returns
% 2.1 Descriptive Statistics for Log Returns
mean_ret = mean(portfolio_returns);
std_ret = std(portfolio_returns);
min_ret = min(portfolio_returns);
max_ret = max(portfolio_returns);
median_ret = median(portfolio_returns);
fprintf('Descriptive Statistics (Log Returns):\nMean: %.4f, Std: %.4f, Median: %.4f, Min: %.4f, Max: %.4f\n',...
    mean_ret, std_ret, median_ret, min_ret, max_ret);

figure; 
histogram(portfolio_returns, 50, 'Normalization', 'pdf');
xlabel('Log Portfolio Returns'); ylabel('Density');
title('Histogram of Log Portfolio Returns');  % Added title
% (Graph 1: Histogram of Log Portfolio Returns)
saveas(gcf, fullfile('image', 'Graph1_Histogram_LogReturns.png'));
close(gcf);

% 2.2 Additional Analysis for Simple Returns
% 2.2.1 Min-Max Scaled Prices Plot
scaled_prices = (prices - min(prices)) ./ (max(prices) - min(prices));
figure; 
plot(dates_all, scaled_prices, 'LineWidth', 1.5);
xlabel('Date'); ylabel('Scaled Price');
legend(tickers, 'Location', 'best');
title('Min-Max Scaled Prices for 6 Stocks');  % Added title
% (Graph 2: Min-Max Scaled Prices for 6 Stocks)
saveas(gcf, fullfile('image', 'Graph2_MinMaxScaled_Prices.png'));
close(gcf);

% 2.2.2 Cumulative Returns and Daily Returns Bar Chart (Simple Returns)
cumulative_returns_simple = cumprod(1 + portfolio_returns_simple);
figure;
subplot(2,1,1);
plot(dates_returns, cumulative_returns_simple, 'LineWidth', 1.5);
xlabel('Date'); ylabel('Cumulative Return');
subplot(2,1,2);
bar(dates_returns, portfolio_returns_simple);
xlabel('Date'); ylabel('Daily Return');
sgtitle('Cumulative and Daily Returns for Simple Returns');  % Added overall title
% (Graph 3: Cumulative and Daily Returns for Simple Returns)
saveas(gcf, fullfile('image', 'Graph3_Cumulative_DailyReturns_Simple.png'));
close(gcf);

% 2.2.3 Overlapped Histogram of Simple Returns with Normal PDF
figure;
histogram(portfolio_returns_simple, 50, 'Normalization','pdf'); hold on;
x_vals_simple = linspace(min(portfolio_returns_simple), max(portfolio_returns_simple), 100);
norm_pdf_simple = normpdf(x_vals_simple, mean(portfolio_returns_simple), std(portfolio_returns_simple));
plot(x_vals_simple, norm_pdf_simple, 'r-', 'LineWidth',2);
xlabel('Simple Portfolio Returns'); ylabel('Density');
legend('Empirical','Normal PDF');
title('Overlapped Histogram of Simple Returns with Normal PDF');  % Added title
% (Graph 4: Overlapped Histogram of Simple Returns with Normal PDF)
saveas(gcf, fullfile('image', 'Graph4_SimpleReturns_Histogram_NormalPDF.png'));
close(gcf);

% 2.3 Overlapped Histogram of Log Returns with Normal PDF
figure;
histogram(portfolio_returns, 50, 'Normalization','pdf'); hold on;
x_vals_log = linspace(min(portfolio_returns), max(portfolio_returns), 100);
norm_pdf_log = normpdf(x_vals_log, mean_ret, std_ret);
plot(x_vals_log, norm_pdf_log, 'r-', 'LineWidth',2);
xlabel('Log Portfolio Returns'); ylabel('Density');
legend('Empirical','Normal PDF');
title('Overlapped Histogram of Log Returns with Normal PDF');  % Added title
% (Graph 5: Overlapped Histogram of Log Returns with Normal PDF)
saveas(gcf, fullfile('image', 'Graph5_LogReturns_Histogram_NormalPDF.png'));
close(gcf);

% 2.4 QQ Plots for Log and Simple Returns
figure;
subplot(1,2,1);
qqplot(portfolio_returns);
xlabel('Theoretical Quantiles'); ylabel('Sample Quantiles');
subplot(1,2,2);
qqplot(portfolio_returns_simple);
xlabel('Theoretical Quantiles'); ylabel('Sample Quantiles');
sgtitle('QQ Plots of Log Returns and Simple Returns');  % Added overall title
% (Graph 6: QQ Plots of Log Returns and Simple Returns)
saveas(gcf, fullfile('image', 'Graph6_QQPlots.png'));
close(gcf);

% 2.5 Jarque-Bera Test for Log Returns
[JB_h, JB_p, JB_stat] = jbtest(portfolio_returns, 0.05);
fprintf('Jarque-Bera test (Log Returns): statistic = %.4f, p-value = %.4f\n', JB_stat, JB_p);

% 2.6 Empirical vs. Theoretical CDF Comparison (Log Returns)
[f_emp, x_emp] = ecdf(portfolio_returns);
f_theo = normcdf(x_emp, mean_ret, std_ret);
figure;
plot(x_emp, f_emp, 'b-', 'LineWidth', 1.5); hold on;
plot(x_emp, f_theo, 'r--', 'LineWidth', 1.5);
xlabel('Log Portfolio Returns'); ylabel('CDF');
legend('Empirical CDF','Theoretical Normal CDF');
title('Empirical vs. Theoretical CDF of Log Returns');  % Added title
% (Graph 7: Empirical vs. Theoretical CDF of Log Returns)
saveas(gcf, fullfile('image', 'Graph7_ECDF_vs_TheoreticalCDF.png'));
close(gcf);

% 2.7 Kolmogorov-Smirnov Test for Log Returns
[h_ks, p_ks, ks_stat] = kstest(portfolio_returns, 'CDF', makedist('Normal','mu',mean_ret,'sigma',std_ret));
fprintf('Kolmogorov-Smirnov test (Log Returns): statistic = %.4f, p-value = %.4f\n', ks_stat, p_ks);

% 2.8 Ljung-Box Test for Autocorrelation of Log Returns
figure;
subplot(3,1,1);
autocorr(portfolio_returns);
xlabel('Lag'); ylabel('ACF');
subplot(3,1,2);
autocorr(abs(portfolio_returns));
xlabel('Lag'); ylabel('ACF');
subplot(3,1,3);
autocorr(portfolio_returns.^2);
xlabel('Lag'); ylabel('ACF');
sgtitle('Autocorrelation of Log Returns, Absolute and Squared Returns');  % Added overall title
% (Graph 8: Autocorrelation of Log Returns, Absolute and Squared Returns)
saveas(gcf, fullfile('image', 'Graph8_Autocorrelation_Log_Abs_Squared.png'));
close(gcf);

%% 3. 6-Month Rolling Window VaR Estimation and Backtesting
confidence_levels = [0.90, 0.99];
var_results = struct();
% VaR methods: HS, Gaussian, MHS, MonteCarlo
var_methods = {'HS', 'Gaussian', 'MHS', 'MonteCarlo'};
start_date_rolling = datetime('2014-07-01');
end_date_analysis = dates_returns(end);
current_date = start_date_rolling;

while current_date <= end_date_analysis
    % Create valid field name: use 'dd_mmm_yyyy' format and replace '-' with '_'
    valid_date = strrep(datestr(current_date, 'dd_mmm_yyyy'), '-', '_');
    if isstrprop(valid_date(1), 'digit')
        valid_date = ['d_' valid_date];
    end
    start_window = current_date - calmonths(6);
    mask_window = (dates_returns >= start_window) & (dates_returns < current_date);
    rolling_returns = portfolio_returns(mask_window);
    
    if numel(rolling_returns) >= 1
        for conf = 1:length(confidence_levels)
            confidence_level = confidence_levels(conf);
            for m = 1:length(var_methods)
                method = var_methods{m};
                if strcmp(method, 'MonteCarlo')
                    % Monte Carlo simulation within the rolling window:
                    mu = mean(rolling_returns);
                    sigma = std(rolling_returns);
                    sim_returns = normrnd(mu, sigma, 10000, 1);
                    VaR_value = -prctile(sim_returns, (1 - confidence_level)*100);
                else
                    VaR_value = calcVaR(rolling_returns, confidence_level, method);
                end
                var_results.(valid_date).(sprintf('%s_%.0f', method, round(confidence_level*100))) = VaR_value;
            end
        end
    else
        for conf = 1:length(confidence_levels)
            for m = 1:length(var_methods)
                var_results.(valid_date).(sprintf('%s_%.0f', var_methods{m}, round(confidence_levels(conf)*100))) = NaN;
            end
        end
    end
    current_date = current_date + days(1);
end

fprintf('6-month rolling window 1-day VaR estimation complete.\n');

% VaR Violation Calculation:
var_violations = struct();
mask_violation = (dates_returns >= start_date_rolling);
violation_returns = portfolio_returns(mask_violation);
violation_dates = dates_returns(mask_violation);

for conf = 1:length(confidence_levels)
    for m = 1:length(var_methods)
        violation_count = 0;
        for k = 1:length(violation_dates)
            valid_date = strrep(datestr(violation_dates(k), 'dd_mmm_yyyy'), '-', '_');
            if isstrprop(valid_date(1), 'digit')
                valid_date = ['d_' valid_date];
            end
            var_key = sprintf('%s_%.0f', var_methods{m}, round(confidence_levels(conf)*100));
            if isfield(var_results, valid_date) && isfield(var_results.(valid_date), var_key)
                VaR_est = var_results.(valid_date).(var_key);
                if ~isnan(VaR_est) && (violation_returns(k) <= -VaR_est)
                    violation_count = violation_count + 1;
                end
            end
        end
        var_violations.(sprintf('%s_%.0f', var_methods{m}, round(confidence_levels(conf)*100))) = violation_count;
    end
end

fprintf('VaR violation count calculation complete.\n');

% Backtesting Performance Evaluation using Kupiec and Conditional Coverage Tests.
alpha_levels = 1 - confidence_levels;
T = length(violation_returns);
results_backtest = struct();
for conf = 1:length(alpha_levels)
    for m = 1:length(var_methods)
        var_key = sprintf('%s_%.0f', var_methods{m}, round(confidence_levels(conf)*100));
        if isfield(var_violations, var_key)
            violations = var_violations.(var_key);
            N = T;
            p_expected = alpha_levels(conf);
            p_actual = violations / T;
            LR_uc = -2 * ( log((1-p_expected)^(N-violations) * p_expected^(violations)) - ...
                           log((1-p_actual)^(N-violations) * p_actual^(violations)) );
            p_value_uc = 1 - chi2cdf(LR_uc, 1);
            results_backtest.(var_key).Kupiec_LR = LR_uc;
            results_backtest.(var_key).Kupiec_p = p_value_uc;
            
            % Conditional Coverage Test using hit sequence.
            hitSeq = zeros(T,1);
            for k = 1:T
                valid_date = strrep(datestr(violation_dates(k), 'dd_mmm_yyyy'), '-', '_');
                if isstrprop(valid_date(1), 'digit')
                    valid_date = ['d_' valid_date];
                end
                if isfield(var_results, valid_date) && isfield(var_results.(valid_date), var_key)
                    VaR_est = var_results.(valid_date).(var_key);
                    if ~isnan(VaR_est) && (violation_returns(k) <= -VaR_est)
                        hitSeq(k) = 1;
                    end
                end
            end
            
            n00 = 0; n01 = 0; n10 = 0; n11 = 0;
            for k = 2:T
                if hitSeq(k-1)==0 && hitSeq(k)==0, n00 = n00 + 1; end
                if hitSeq(k-1)==0 && hitSeq(k)==1, n01 = n01 + 1; end
                if hitSeq(k-1)==1 && hitSeq(k)==0, n10 = n10 + 1; end
                if hitSeq(k-1)==1 && hitSeq(k)==1, n11 = n11 + 1; end
            end
            
            if (n00+n01)>0 && (n10+n11)>0
                pi0 = n01 / (n00+n01);
                pi1 = n11 / (n10+n11);
                pi = (n01+n11) / (n00+n01+n10+n11);
                LR_ind = -2 * ( log(pi^(n01+n11) * (1-pi)^(n00+n10)) - ...
                                log(pi0^n01 * (1-pi0)^n00 * pi1^n11 * (1-pi1)^n10) );
                LR_cc = LR_uc + LR_ind;
                p_value_cc = 1 - chi2cdf(LR_cc, 2);
                results_backtest.(var_key).ConditionalCoverage_LR = LR_cc;
                results_backtest.(var_key).ConditionalCoverage_p = p_value_cc;
            else
                results_backtest.(var_key).ConditionalCoverage_LR = NaN;
                results_backtest.(var_key).ConditionalCoverage_p = NaN;
            end
            results_backtest.(var_key).Distributional_Tests = 'Not performed.';
        else
            results_backtest.(var_key).Kupiec_LR = NaN;
            results_backtest.(var_key).Kupiec_p = NaN;
            results_backtest.(var_key).ConditionalCoverage_LR = NaN;
            results_backtest.(var_key).ConditionalCoverage_p = NaN;
            results_backtest.(var_key).Distributional_Tests = 'VaR results not available for backtesting.';
        end
    end
end

fprintf('Backtesting evaluation complete.\n');

%% 4. Additional Section: 120-Day Rolling Window VaR Estimation
window = 120;         % 120 trading days window
n = 1;                % 1-day horizon
confidence_interval = linspace(0.01, 0.99, 100);
n_iterations = length(portfolio_returns) - window + 1;

% Initialize matrices for four methods.
Var_Parametric = zeros(n_iterations, numel(confidence_interval));
Var_NonParametric = zeros(n_iterations, numel(confidence_interval));
Var_MHS = zeros(n_iterations, numel(confidence_interval));
Var_MonteCarlo = zeros(n_iterations, numel(confidence_interval));

tic;
for j = 1:numel(confidence_interval)
    conf_level = confidence_interval(j);
    for i = 1:n_iterations
        iter_data = portfolio_returns(i:i+window-1);
        Var_Parametric(i, j) = calcVaR(iter_data, conf_level, 'Gaussian');
        Var_NonParametric(i, j) = calcVaR(iter_data, conf_level, 'HS');
        Var_MHS(i, j) = calcVaR(iter_data, conf_level, 'MHS');
        % Monte Carlo VaR:
        mu = mean(iter_data);
        SD = std(iter_data);
        k = mu * window - ((SD * sqrt(window))^2)/2;
        sim_returns = normrnd(0,1,500,1) * sqrt(n/250) * (SD * sqrt(window)) + k * n/250;
        VaR_mc = -prctile(sim_returns, (1-conf_level)*100);
        Var_MonteCarlo(i, j) = VaR_mc;
    end 
end
toc;

% Plot: VaRs Over Time at 90% Confidence Level for 120-Day Window
VaR_level_index = 90; % 90% confidence.
additional_rows = zeros(window, 1);
Var_Parametric_plot = [additional_rows; Var_Parametric(:, VaR_level_index)];
Var_NonParametric_plot = [additional_rows; Var_NonParametric(:, VaR_level_index)];
Var_MHS_plot = [additional_rows; Var_MHS(:, VaR_level_index)];
Var_MonteCarlo_plot = [additional_rows; Var_MonteCarlo(:, VaR_level_index)];

figure;
plot(portfolio_returns, 'LineWidth', 0.25); hold on;
plot(-Var_Parametric_plot, 'LineWidth', 1.5, 'LineStyle', '-.');
plot(-Var_NonParametric_plot, 'LineWidth', 1.5, 'LineStyle', '--');
plot(-Var_MHS_plot, 'LineWidth', 1.5, 'LineStyle', ':');
plot(-Var_MonteCarlo_plot, 'LineWidth', 1, 'LineStyle', '-');
xlabel('Time (Days)','Interpreter','latex')
ylabel('VaR / Portfolio Returns','Interpreter','latex')
legend('Portfolio Returns','Parametric VaR','Non-Parametric VaR','Modified HS VaR','Monte Carlo VaR', 'Interpreter','latex')
title('120-Day Rolling VaRs Over Time at 90% Confidence');  % Added title
% (Graph 9: 120-Day Rolling VaRs Over Time at 90% Confidence)
saveas(gcf, fullfile('image', 'Graph9_120DayRolling_VaR_90.png'));
close(gcf);

% Plot: VaR vs Confidence Level Comparison (for the first iteration) for 120-Day Window
figure;
subplot(2,2,1)
plot(confidence_interval, Var_Parametric(1,:), 'LineWidth', 0.5);
xlabel('Confidence Level'); ylabel('VaR'); grid on
subplot(2,2,2)
plot(confidence_interval, Var_NonParametric(1,:), 'LineWidth', 0.5);
xlabel('Confidence Level'); ylabel('VaR'); grid on
subplot(2,2,3)
plot(confidence_interval, Var_MHS(1,:), 'LineWidth', 0.5);
xlabel('Confidence Level'); ylabel('VaR'); grid on
subplot(2,2,4)
plot(confidence_interval, Var_MonteCarlo(1,:), 'LineWidth', 0.5);
xlabel('Confidence Level'); ylabel('VaR'); grid on
sgtitle('VaR vs Confidence Level Comparison for 120-Day Window');  % Added overall title
% (Graph 10: VaR vs Confidence Level Comparison for 120-Day Window)
saveas(gcf, fullfile('image', 'Graph10_VaR_vs_Confidence_120Day.png'));
close(gcf);

%% 5. New Additions

% 5A. Enlarged Histogram of the Tail Region of Log Returns
tail_threshold = mean_ret - 2*std_ret;  % example threshold
tail_data = portfolio_returns(portfolio_returns < tail_threshold);
figure;
histogram(tail_data, 50, 'Normalization','pdf');
xlabel('Log Returns (Tail Region)'); ylabel('Density');
title('Enlarged Histogram of Tail Region of Log Returns');  % Added title
% (Graph 11: Enlarged Histogram of Tail Region of Log Returns)
saveas(gcf, fullfile('image', 'Graph11_TailHistogram_LogReturns.png'));
close(gcf);

% 5B. Separate Autocorrelation Plots for Absolute and Squared Returns
figure;
autocorr(abs(portfolio_returns));
xlabel('Lag'); ylabel('ACF');
title('Autocorrelation of Absolute Log Returns');  % Added title
% (Graph 12: Autocorrelation of Absolute Log Returns)
saveas(gcf, fullfile('image', 'Graph12_ACF_AbsoluteReturns.png'));
close(gcf);

figure;
autocorr(portfolio_returns.^2);
xlabel('Lag'); ylabel('ACF');
title('Autocorrelation of Squared Log Returns');  % Added title
% (Graph 13: Autocorrelation of Squared Log Returns)
saveas(gcf, fullfile('image', 'Graph13_ACF_SquaredReturns.png'));
close(gcf);

% 5C. Conditional Coverage Test Results Tables (Christoffersen's CC Test)
kupiec_fields = fieldnames(results_backtest);
% For 90% Confidence:
ccData90 = [];
MethodNames_cc90 = {};
for i = 1:length(kupiec_fields)
    key = kupiec_fields{i};
    if ~isempty(regexp(key, '_90$', 'once'))
        ccData90 = [ccData90; results_backtest.(key).ConditionalCoverage_LR, results_backtest.(key).ConditionalCoverage_p];
        MethodNames_cc90{end+1} = key;
    end
end
CC_Table90 = array2table(ccData90, 'VariableNames', {'CC_LR', 'CC_p'});
CC_Table90.Properties.RowNames = MethodNames_cc90;
disp('Conditional Coverage Test Results (90% Confidence):');
disp(CC_Table90);
% (Graph 14: Conditional Coverage Test Results Table for 90% Confidence)

% For 99% Confidence:
ccData99 = [];
MethodNames_cc99 = {};
for i = 1:length(kupiec_fields)
    key = kupiec_fields{i};
    if ~isempty(regexp(key, '_99$', 'once'))
        ccData99 = [ccData99; results_backtest.(key).ConditionalCoverage_LR, results_backtest.(key).ConditionalCoverage_p];
        MethodNames_cc99{end+1} = key;
    end
end
CC_Table99 = array2table(ccData99, 'VariableNames', {'CC_LR', 'CC_p'});
CC_Table99.Properties.RowNames = MethodNames_cc99;
disp('Conditional Coverage Test Results (99% Confidence):');
disp(CC_Table99);
% (Graph 15: Conditional Coverage Test Results Table for 99% Confidence)

% 5D. Probability Integral Transform (PIT) Analysis
% Compute PIT based on Gaussian assumption for log returns.
pit = normcdf(portfolio_returns, mean_ret, std_ret);
[h_pit, pval_pit] = kstest(pit, 'CDF', makedist('Uniform',0,1));
fprintf('PIT Kolmogorov-Smirnov Test: h = %d, p-value = %.4f\n', h_pit, pval_pit);
figure;
histogram(pit, 50, 'Normalization','pdf'); hold on;
x_uni = linspace(0,1,100);
plot(x_uni, ones(size(x_uni)), 'r-', 'LineWidth',2);
xlabel('PIT Value'); ylabel('Density');
title('Histogram of PIT with Uniform Density Overlay');  % Added title
% (Graph 16: Histogram of PIT with Uniform Density Overlay)
saveas(gcf, fullfile('image', 'Graph16_PIT_Histogram.png'));
close(gcf);

% 5E. VaR Violation Clustering Analysis
% For demonstration, use 'Gaussian_90' violations.
violation_indicator = zeros(length(violation_dates),1);
for k = 1:length(violation_dates)
    valid_date = strrep(datestr(violation_dates(k), 'dd_mmm_yyyy'), '-', '_');
    if isstrprop(valid_date(1), 'digit')
        valid_date = ['d_' valid_date];
    end
    var_key = 'Gaussian_90';
    if isfield(var_results, valid_date) && isfield(var_results.(valid_date), var_key)
        VaR_est = var_results.(valid_date).(var_key);
        if ~isnan(VaR_est) && (violation_returns(k) <= -VaR_est)
            violation_indicator(k) = 1;
        end
    end
end
figure;
stem(violation_dates, violation_indicator, 'filled');
xlabel('Date'); ylabel('Violation (1 if occurred)');
title('VaR Violation Clustering Stem Plot');  % Added title
% (Graph 17: VaR Violation Clustering Stem Plot)
saveas(gcf, fullfile('image', 'Graph17_ViolationClustering.png'));
close(gcf);

%% 6. End of Script

%% Function: calcVaR
function VaR = calcVaR(returnData, confLevel, method)
% calcVaR calculates a single VaR value for a given vector of returns,
% a given confidence level, and the specified method.
% Inputs:
%   returnData - Vector of return data.
%   confLevel  - Confidence level (e.g., 0.90 or 0.99).
%   method     - VaR calculation method: 'HS', 'Gaussian', 'MHS', or 'MonteCarlo'.
% Output:
%   VaR        - The calculated VaR value (scalar).
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
        % Modified Historical Simulation with EWMA weighting.
        T = length(returnData);
        lambda = 0.94;
        weights = lambda.^( (T-1):-1:0 )';
        weights = weights / sum(weights);
        [sortedData, sortIdx] = sort(returnData);
        sortedWeights = weights(sortIdx);
        cumWeights = cumsum(sortedWeights);
        idx = find(cumWeights >= alpha, 1, 'first');
        VaR = -sortedData(idx);
    case 'MonteCarlo'
        % Monte Carlo simulation approach (10,000 simulations by default).
        mu = mean(returnData);
        sigma = std(returnData);
        sim_returns = normrnd(mu, sigma, 10000, 1);
        VaR = -prctile(sim_returns, (1-confLevel)*100);
    otherwise
        VaR = NaN;
end
end
