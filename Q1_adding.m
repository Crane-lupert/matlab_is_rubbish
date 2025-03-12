%% Group Coursework - Q1: VAR Modelling (Integrated Revised Version with calcVaR Function)
% This script performs the following tasks:
% 1. Data Preparation and Portfolio Returns Calculation:
%    - Reads adjusted closing prices from an Excel file.
%    - Constructs dates using numeric month data.
%    - Computes daily log returns and simple returns, and builds an equally weighted portfolio.
%
% 2. Statistical Analysis of Portfolio Returns:
%    - Computes descriptive statistics and displays a histogram.
%    - Additional analyses include:
%         a) Min-Max scaled price plot.
%         b) Cumulative returns and daily returns (using simple returns) bar chart.
%         c) Overlapped histogram of simple returns with a fitted Normal PDF,
%            with extreme losses highlighted and skewness annotated.
%         d) Overlapped histogram of log returns with Normal PDF,
%            with extreme losses highlighted and skewness annotated.
%         e) QQ Plot for log returns and simple returns with skewness annotation.
%         f) Skewness and Kurtosis values for both return types are printed.
%         g) Jarque-Bera test, empirical vs theoretical CDF, Kolmogorov-Smirnov test.
%         h) Ljung-Box test for autocorrelation.
%         i) Variance Ratio Test.
%
% 3. 6-Month Rolling Window VaR Estimation and Backtesting:
%    - Estimates daily VaR at 90% and 99% confidence levels using four methods:
%         a) Historical Simulation (HS)
%         b) Parametric Gaussian VaR
%         c) Modified Historical Simulation (MHS) using EWMA weighting
%         d) Delta-Normal Approximation
%    - Computes VaR violations and evaluates backtesting performance.
%
% 4. Additional Section: 120-Day Rolling Window VaR Estimation:
%    - Estimates VaR using a 120-day window via Parametric (Gaussian), Non-Parametric (HS),
%      Modified HS, and Monte Carlo methods.
%    - Plots VaRs over time at 90% confidence and compares VaR vs. confidence levels.
%
% Note: All comments and code are in English.
%       Adjust file paths and parameters as needed.

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

figure('Color',[1 1 1]);
histogram(portfolio_returns, 50, 'Normalization', 'pdf');
xlabel('Log Portfolio Returns'); ylabel('Density');
title('Histogram of Log Portfolio Returns');

% 2.2 Additional Analysis for Simple Returns

% 2.2.1 Min-Max Scaled Prices Plot
scaled_prices = (prices - min(prices)) ./ (max(prices) - min(prices));
figure('Color',[1 1 1]);
plot(dates_all, scaled_prices, 'LineWidth', 1.5);
xlabel('Date'); ylabel('Scaled Price');
title('Min-Max Scaled Prices for 6 Stocks');
legend(tickers, 'Location', 'best');

% 2.2.2 Cumulative Returns and Daily Returns Bar Chart (Simple Returns)
cumulative_returns_simple = cumprod(1 + portfolio_returns_simple);
figure('Color',[1 1 1]);
subplot(2,1,1);
plot(dates_returns, cumulative_returns_simple, 'LineWidth', 1.5);
xlabel('Date'); ylabel('Cumulative Return');
title('Cumulative Portfolio Returns (Simple Returns)');
subplot(2,1,2);
bar(dates_returns, portfolio_returns_simple);
xlabel('Date'); ylabel('Daily Return');
title('Daily Portfolio Returns (Simple Returns)');

% 2.2.3 Overlapped Histogram of Simple Returns with Normal PDF
figure('Color',[1 1 1]);
histogram(portfolio_returns_simple, 50, 'Normalization','pdf'); hold on;
x_vals_simple = linspace(min(portfolio_returns_simple), max(portfolio_returns_simple), 100);
norm_pdf_simple = normpdf(x_vals_simple, mean(portfolio_returns_simple), std(portfolio_returns_simple));
plot(x_vals_simple, norm_pdf_simple, 'r-', 'LineWidth',2);
xlabel('Simple Portfolio Returns'); ylabel('Density');
title('Overlapped Histogram of Simple Returns with Normal PDF');
legend('Empirical','Normal PDF');
% Annotate skewness for simple returns
skew_simple = skewness(portfolio_returns_simple);
text(mean(portfolio_returns_simple), max(norm_pdf_simple)*0.9, sprintf('Skewness = %.4f', skew_simple), 'FontSize',12, 'Color','k');

% 2.3 Overlapped Histogram of Log Returns with Normal PDF
figure('Color',[1 1 1]);
histogram(portfolio_returns, 50, 'Normalization','pdf'); hold on;
x_vals_log = linspace(min(portfolio_returns), max(portfolio_returns), 100);
norm_pdf_log = normpdf(x_vals_log, mean_ret, std_ret);
plot(x_vals_log, norm_pdf_log, 'r-', 'LineWidth',2);
xlabel('Log Portfolio Returns'); ylabel('Density');
title('Overlapped Histogram of Log Returns with Normal PDF');
legend('Empirical','Normal PDF');
% Annotate skewness for log returns
skew_log = skewness(portfolio_returns);
text(mean(portfolio_returns), max(norm_pdf_log)*0.9, sprintf('Skewness = %.4f', skew_log), 'FontSize',12, 'Color','k');

% 2.4 QQ Plot for Log Returns and Simple Returns with Annotation
figure('Color',[1 1 1]);
subplot(1,2,1);
qqplot(portfolio_returns);
title('QQ Plot of Log Returns');
text(min(portfolio_returns), max(portfolio_returns), sprintf('Skewness = %.4f', skew_log), 'FontSize',12, 'Color','b');
subplot(1,2,2);
qqplot(portfolio_returns_simple);
title('QQ Plot of Simple Returns');
text(min(portfolio_returns_simple), max(portfolio_returns_simple), sprintf('Skewness = %.4f', skew_simple), 'FontSize',12, 'Color','b');
% Note: QQ plots for simple and log returns show similar patterns; the transformation
% to log returns is primarily motivated by economic theory to better capture percentage changes.

% 2.5 Jarque-Bera Test for Log Returns
[JB_h, JB_p, JB_stat] = jbtest(portfolio_returns, 0.05);
fprintf('Jarque-Bera test (Log Returns): statistic = %.4f, p-value = %.4f\n', JB_stat, JB_p);

% 2.6 Empirical vs. Theoretical CDF Comparison (Log Returns)
[f_emp, x_emp] = ecdf(portfolio_returns);
f_theo = normcdf(x_emp, mean_ret, std_ret);
figure('Color',[1 1 1]);
plot(x_emp, f_emp, 'b-', 'LineWidth', 1.5); hold on;
plot(x_emp, f_theo, 'r--', 'LineWidth', 1.5);
xlabel('Log Portfolio Returns'); ylabel('CDF');
title('Empirical vs. Theoretical CDF of Log Returns');
legend('Empirical CDF','Theoretical Normal CDF');

% 2.7 Kolmogorov-Smirnov Test for Log Returns
[h_ks, p_ks, ks_stat] = kstest(portfolio_returns, 'CDF', makedist('Normal','mu',mean_ret,'sigma',std_ret));
fprintf('Kolmogorov-Smirnov test (Log Returns): statistic = %.4f, p-value = %.4f\n', ks_stat, p_ks);

% 2.8 Ljung-Box Test for Autocorrelation (Log Returns)
figure('Color',[1 1 1]);
subplot(3,1,1);
autocorr(portfolio_returns);
title('Autocorrelation of Log Returns');
subplot(3,1,2);
autocorr(abs(portfolio_returns));
title('Autocorrelation of Absolute Log Returns');
subplot(3,1,3);
autocorr(portfolio_returns.^2);
title('Autocorrelation of Squared Log Returns');

% 2.9 Variance Ratio Test (if available)
if exist('vratiotest','file')
    [h_vr, p_vr, stat_vr] = vratiotest(portfolio_returns);
    fprintf('Variance Ratio Test (Log Returns): statistic = %.4f, p-value = %.4f\n', stat_vr, p_vr);
else
    fprintf('Variance Ratio Test function (vratiotest) not available.\n');
end

%% 3. 6-Month Rolling Window VaR Estimation and Backtesting
confidence_levels = [0.90, 0.99];
var_results = struct();
var_methods = {'HS', 'Gaussian', 'MHS', 'DeltaNorm'};
start_date_rolling = datetime('2014-07-01');
end_date_analysis = dates_returns(end);
current_date = start_date_rolling;

while current_date <= end_date_analysis
    valid_date = matlab.lang.makeValidName(datestr(current_date));
    start_window = current_date - calmonths(6);
    mask_window = (dates_returns >= start_window) & (dates_returns < current_date);
    rolling_returns = portfolio_returns(mask_window);
    
    if numel(rolling_returns) >= 1
        for conf = 1:length(confidence_levels)
            confidence_level = confidence_levels(conf);
            for m = 1:length(var_methods)
                method = var_methods{m};
                VaR_value = calcVaR(rolling_returns, confidence_level, method);
                var_results.(valid_date).(sprintf('%s_%d', method, confidence_level*100)) = VaR_value;
            end
        end
    else
        for conf = 1:length(confidence_levels)
            for m = 1:length(var_methods)
                var_results.(valid_date).(sprintf('%s_%d', var_methods{m}, confidence_levels(conf)*100)) = NaN;
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

disp('Sample of actual returns and corresponding dates:');
disp(table(violation_dates(1:5), violation_returns(1:5)));

for conf = 1:length(confidence_levels)
    for m = 1:length(var_methods)
        violation_count = 0;
        for k = 1:length(violation_dates)
            valid_date = matlab.lang.makeValidName(datestr(violation_dates(k)));
            var_key = sprintf('%s_%d', var_methods{m}, confidence_levels(conf)*100);
            if isfield(var_results, valid_date) && isfield(var_results.(valid_date), var_key)
                VaR_est = var_results.(valid_date).(var_key);
                if ~isnan(VaR_est) && (violation_returns(k) <= -VaR_est)
                    violation_count = violation_count + 1;
                end
            end
        end
        var_violations.(sprintf('%s_%d', var_methods{m}, confidence_levels(conf)*100)) = violation_count;
    end
end

fprintf('VaR violation count calculation complete.\n');
disp('Results stored in var_violations struct:');
disp(var_violations);

% Backtesting Performance Evaluation using Kupiec and Conditional Coverage Tests.
alpha_levels = 1 - confidence_levels;
T = length(violation_returns);
results_backtest = struct();

for conf = 1:length(alpha_levels)
    for m = 1:length(var_methods)
        var_key = sprintf('%s_%d', var_methods{m}, confidence_levels(conf)*100);
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
            
            hitSeq = zeros(T,1);
            for k = 1:T
                valid_date = matlab.lang.makeValidName(datestr(violation_dates(k)));
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
disp('Backtesting results stored in results_backtest struct:');
disp(results_backtest);

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
        % Use calcVaR function for each method.
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

% Plot: VaRs Over Time at 90% Confidence Level
VaR_level_index = 90; % 90% confidence.
additional_rows = zeros(window, 1);
Var_Parametric_plot = [additional_rows; Var_Parametric(:, VaR_level_index)];
Var_NonParametric_plot = [additional_rows; Var_NonParametric(:, VaR_level_index)];
Var_MHS_plot = [additional_rows; Var_MHS(:, VaR_level_index)];
Var_MonteCarlo_plot = [additional_rows; Var_MonteCarlo(:, VaR_level_index)];

figure('Color',[1 1 1])
plot(portfolio_returns, 'LineWidth', 0.25); hold on;
plot(-Var_Parametric_plot, 'LineWidth', 1.5, 'LineStyle', '-.');
plot(-Var_NonParametric_plot, 'LineWidth', 1.5, 'LineStyle', '--');
plot(-Var_MHS_plot, 'LineWidth', 1.5, 'LineStyle', ':');
plot(-Var_MonteCarlo_plot, 'LineWidth', 1, 'LineStyle', '-');
xlabel('Time (Days)','Interpreter','latex')
ylabel('VaR / Portfolio Returns','Interpreter','latex')
legend('Portfolio Returns','Parametric VaR','Non-Parametric VaR','Modified HS VaR','Monte Carlo VaR', 'Interpreter','latex')
title('VaRs Over Time at 90% Confidence','Interpreter','latex')
grid on;
hold off;

% Plot: VaR vs Confidence Level Comparison (for the first iteration)
figure('Color',[1 1 1])
subplot(2,2,1)
plot(confidence_interval, Var_Parametric(1,:), 'LineWidth', 0.5);
title('Parametric VaR vs Confidence','Interpreter','latex')
xlabel('Confidence Level','Interpreter','latex')
ylabel('VaR','Interpreter','latex')
grid on

subplot(2,2,2)
plot(confidence_interval, Var_NonParametric(1,:), 'LineWidth', 0.5);
title('Non-Parametric VaR vs Confidence','Interpreter','latex')
xlabel('Confidence Level','Interpreter','latex')
ylabel('VaR','Interpreter','latex')
grid on

subplot(2,2,3)
plot(confidence_interval, Var_MHS(1,:), 'LineWidth', 0.5);
title('Modified HS VaR vs Confidence','Interpreter','latex')
xlabel('Confidence Level','Interpreter','latex')
ylabel('VaR','Interpreter','latex')
grid on

subplot(2,2,4)
plot(confidence_interval, Var_MonteCarlo(1,:), 'LineWidth', 0.5);
title('Monte Carlo VaR vs Confidence','Interpreter','latex')
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
