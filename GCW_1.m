%% Group Coursework - Q1: VAR Modelling (Integrated Version)
% This script performs the following tasks:
% 1. Data Preparation and Portfolio Returns Calculation:
%    - Reads adjusted closing prices for six stocks from an Excel file.
%    - Constructs dates using numeric month data.
%    - Computes daily log returns and builds an equally weighted portfolio.
%
% 2. Statistical Analysis of Portfolio Returns:
%    - Computes descriptive statistics and displays a histogram.
%
% 3. 6-Month Rolling Window VaR Estimation and Backtesting:
%    - Estimates daily VaR at 90% and 99% confidence levels using four methods:
%      a) Historical Simulation (HS)
%      b) Parametric Gaussian VaR
%      c) Modified Historical Simulation (MHS)
%      d) Delta-Normal Approximation (as an alternative to t-distribution)
%
% 4. VaR Violation Calculation and Backtesting Evaluation:
%    - Computes the number of VaR violations for each model and each confidence level.
%    - Evaluates backtesting performance using the Kupiec and Conditional Coverage tests.
%
% 5. Additional Section: 120-Day Rolling Window VaR Estimation:
%    - Estimates VaR using a 120-day window via Parametric, Non-Parametric, and Monte Carlo methods.
%    - Plots VaR over time and compares VaR values against confidence levels.
%
% Note: Adjust file paths and parameters as needed.

%% 1. Data Preparation and Portfolio Returns Calculation
clear; close all; clc;
format short;

% Load data from the Excel file
filename = 'big_6_adj_closing_prices.xlsx';
dataTable = readtable(filename);

% Rename columns for easier access
% Expected columns: Month, Day, Year, Alphabet, Amazon, Apple, IBM, Microsoft, Nvidia, dollar
dataTable.Properties.VariableNames = {'Month', 'Day', 'Year', 'Alphabet', 'Amazon', 'Apple', 'IBM', 'Microsoft', 'Nvidia', 'dollar'};

% Since the Month column now contains numeric data, we use it directly.
% Construct date strings using numeric month, day, and year (e.g., '9-30-2014')
datesStr = strcat(string(dataTable.Month), '-', string(dataTable.Day), '-', string(dataTable.Year));
datesStr = regexprep(datesStr, '\s+', ' ');
datesStr = strtrim(datesStr);
dates_all = datetime(datesStr, 'InputFormat', 'M-d-yyyy');

% Sort data by date in ascending order
[dates_all, sortIdx] = sort(dates_all);
dataTable = dataTable(sortIdx, :);

% Extract adjusted closing prices for selected stocks:
% Stocks: AAPL (Apple), MSFT (Microsoft), IBM, Nvidia, Alphabet, Amazon.
tickers = {'Alphabet', 'Amazon', 'Apple', 'IBM', 'Microsoft', 'Nvidia'};
prices = table2array(dataTable(:, tickers));

% Compute daily log returns (vectorized)
log_returns = diff(log(prices));
dates_returns = dates_all(2:end);

% Build equally weighted portfolio returns
num_stocks = numel(tickers);
portfolio_weights = ones(1, num_stocks) / num_stocks;
portfolio_returns = log_returns * portfolio_weights';

fprintf('Portfolio returns calculation complete.\n');

%% 2. Statistical Analysis of Portfolio Returns
% Compute descriptive statistics and display histogram
mean_ret = mean(portfolio_returns);
std_ret = std(portfolio_returns);
min_ret = min(portfolio_returns);
max_ret = max(portfolio_returns);
median_ret = median(portfolio_returns);
fprintf('Descriptive Statistics:\nMean: %.4f, Std: %.4f, Median: %.4f, Min: %.4f, Max: %.4f\n',...
    mean_ret, std_ret, median_ret, min_ret, max_ret);

figure('Color',[1 1 1]);
histogram(portfolio_returns, 50, 'Normalization', 'pdf');
xlabel('Portfolio Returns'); ylabel('Density');
title('Histogram of Portfolio Returns');

%% 3. 6-Month Rolling Window VaR Estimation and Backtesting
% Define confidence levels and initialize structure to store VaR estimates.
confidence_levels = [0.90, 0.99];
var_results = struct();
var_methods = {'HS', 'Gaussian', 'MHS', 'DeltaNorm'}; % Methods: Historical, Gaussian, Modified HS, Delta-Normal

% Analysis window: rolling window starts on July 1, 2014
start_date_rolling = datetime('2014-07-01');
end_date_analysis = dates_returns(end);
current_date = start_date_rolling;

while current_date <= end_date_analysis
    % Create a valid field name from the current date
    valid_date = matlab.lang.makeValidName(datestr(current_date));
    
    % Define 6-month rolling window: from (current_date - 6 months) to current_date
    start_window = current_date - calmonths(6);
    mask_window = (dates_returns >= start_window) & (dates_returns < current_date);
    rolling_returns = portfolio_returns(mask_window);
    
    if numel(rolling_returns) >= 1
        for conf = 1:length(confidence_levels)
            confidence_level = confidence_levels(conf);
            alpha = 1 - confidence_level;
            quantile_level = alpha * 100;
            
            % Historical Simulation (HS)
            var_hs = -prctile(rolling_returns, quantile_level);
            var_results.(valid_date).(sprintf('HS_%d', confidence_level*100)) = var_hs;
            
            % Parametric Gaussian VaR
            mu = mean(rolling_returns);
            sigma = std(rolling_returns);
            z_value = norminv(alpha);
            var_gaussian = -(mu + z_value * sigma);
            var_results.(valid_date).(sprintf('Gaussian_%d', confidence_level*100)) = var_gaussian;
            
            % Modified Historical Simulation (MHS)
            var_mhs = -prctile(rolling_returns, quantile_level);
            var_results.(valid_date).(sprintf('MHS_%d', confidence_level*100)) = var_mhs;
            
            % Delta-Normal Approximation (alternative to t-distribution)
            % Assuming linear exposures for bonds and stock prices.
            var_delta = -(mu + z_value * sigma);
            var_results.(valid_date).(sprintf('DeltaNorm_%d', confidence_level*100)) = var_delta;
        end
    else
        for conf = 1:length(confidence_levels)
            confidence_level = confidence_levels(conf);
            for m = 1:length(var_methods)
                var_results.(valid_date).(sprintf('%s_%d', var_methods{m}, confidence_level*100)) = NaN;
            end
        end
    end
    current_date = current_date + days(1);
end

fprintf('6-month rolling window 1-day VaR estimation complete.\n');

% VaR Violation Calculation
% A violation is counted if the actual return is less than or equal to the negative VaR forecast.
var_violations = struct();
mask_violation = (dates_returns >= start_date_rolling);
violation_returns = portfolio_returns(mask_violation);
violation_dates = dates_returns(mask_violation);

disp('Sample of actual returns and corresponding dates:');
disp(table(violation_dates(1:5), violation_returns(1:5)));

for conf = 1:length(confidence_levels)
    confidence_level = confidence_levels(conf);
    for m = 1:length(var_methods)
        method = var_methods{m};
        violation_count = 0;
        for k = 1:length(violation_dates)
            current_date_str = datestr(violation_dates(k));
            valid_date = matlab.lang.makeValidName(current_date_str);
            var_key = sprintf('%s_%d', method, confidence_level*100);
            if isfield(var_results, valid_date) && isfield(var_results.(valid_date), var_key)
                var_forecast = var_results.(valid_date).(var_key);
                if ~isnan(var_forecast) && (violation_returns(k) <= -var_forecast)
                    violation_count = violation_count + 1;
                end
            end
        end
        var_violations.(sprintf('%s_%d', method, confidence_level*100)) = violation_count;
    end
end

fprintf('VaR violation count calculation complete.\n');
disp('Results stored in var_violations struct:');
disp(var_violations);

% Backtesting Performance Evaluation using Kupiec and Conditional Coverage Tests
alpha_levels = 1 - confidence_levels;
T = length(violation_returns);
results_backtest = struct();

for conf = 1:length(alpha_levels)
    alpha = alpha_levels(conf);
    for m = 1:length(var_methods)
        method = var_methods{m};
        var_key = sprintf('%s_%d', method, confidence_levels(conf)*100);
        if isfield(var_violations, var_key)
            violations = var_violations.(var_key);
            N = T;
            p_expected = alpha;
            p_actual = violations / N;
            LR_uc = -2 * ( log((1 - p_expected)^(N - violations) * p_expected^(violations)) - ...
                           log((1 - p_actual)^(N - violations) * p_actual^(violations)) );
            p_value_uc = 1 - chi2cdf(LR_uc, 1);
            results_backtest.(var_key).Kupiec_LR_Statistic = LR_uc;
            results_backtest.(var_key).Kupiec_p_value = p_value_uc;
            
            % Conditional Coverage Test using hit sequence
            hit_sequence = zeros(size(violation_returns));
            for k = 1:length(violation_returns)
                current_date_str = datestr(violation_dates(k));
                valid_date = matlab.lang.makeValidName(current_date_str);
                if isfield(var_results, valid_date) && isfield(var_results.(valid_date), var_key)
                    var_forecast = var_results.(valid_date).(var_key);
                    if ~isnan(var_forecast) && (violation_returns(k) <= -var_forecast)
                        hit_sequence(k) = 1;
                    end
                end
            end
            
            n00 = 0; n01 = 0; n10 = 0; n11 = 0;
            for k = 2:length(hit_sequence)
                if hit_sequence(k-1)==0 && hit_sequence(k)==0, n00 = n00 + 1; end
                if hit_sequence(k-1)==0 && hit_sequence(k)==1, n01 = n01 + 1; end
                if hit_sequence(k-1)==1 && hit_sequence(k)==0, n10 = n10 + 1; end
                if hit_sequence(k-1)==1 && hit_sequence(k)==1, n11 = n11 + 1; end
            end
            
            if (n00 + n01) > 0 && (n10 + n11) > 0
                pi0 = n01 / (n00 + n01);
                pi1 = n11 / (n10 + n11);
                pi = (n01 + n11) / (n00 + n01 + n10 + n11);
                LR_ind = -2 * ( log(pi^(n01+n11) * (1-pi)^(n00+n10)) - ...
                                log(pi0^n01 * (1-pi0)^n00 * pi1^n11 * (1-pi1)^n10) );
                p_value_ind = 1 - chi2cdf(LR_ind, 1);
                LR_cc = LR_uc + LR_ind;
                p_value_cc = 1 - chi2cdf(LR_cc, 2);
                results_backtest.(var_key).Independence_LR_Statistic = LR_ind;
                results_backtest.(var_key).Independence_p_value = p_value_ind;
                results_backtest.(var_key).ConditionalCoverage_LR_Statistic = LR_cc;
                results_backtest.(var_key).ConditionalCoverage_p_value = p_value_cc;
            else
                results_backtest.(var_key).Independence_LR_Statistic = NaN;
                results_backtest.(var_key).Independence_p_value = NaN;
                results_backtest.(var_key).ConditionalCoverage_LR_Statistic = NaN;
                results_backtest.(var_key).ConditionalCoverage_p_value = NaN;
            end
            
            results_backtest.(var_key).Distributional_Tests = 'Not performed.';
        else
            results_backtest.(var_key).Kupiec_LR_Statistic = NaN;
            results_backtest.(var_key).Kupiec_p_value = NaN;
            results_backtest.(var_key).Independence_LR_Statistic = NaN;
            results_backtest.(var_key).Independence_p_value = NaN;
            results_backtest.(var_key).ConditionalCoverage_LR_Statistic = NaN;
            results_backtest.(var_key).ConditionalCoverage_p_value = NaN;
            results_backtest.(var_key).Distributional_Tests = 'VaR results not available for backtesting.';
        end
    end
end

fprintf('Backtesting evaluation complete.\n');
disp('Backtesting results stored in results_backtest struct:');
disp(results_backtest);

%% 4. Additional Section: 120-Day Rolling Window VaR Estimation
% This section calculates VaR estimates using a 120-day window for a range of confidence levels.
window = 120;         % 120 trading days window
n = 1;                % 1-day horizon
confidence_interval = linspace(0.01, 0.99, 100);
n_iterations = length(portfolio_returns) - window + 1;

Var_Parametric = zeros(n_iterations, numel(confidence_interval));
Var_NonParametric = zeros(n_iterations, numel(confidence_interval));
Var_MonteCarlo = zeros(n_iterations, numel(confidence_interval));

tic;
for j = 1:numel(confidence_interval)
    conf_level = confidence_interval(j);
    for i = 1:n_iterations
        iter_data = portfolio_returns(i:i+window-1);
        mu = mean(iter_data) * n;
        SD = std(iter_data) * sqrt(n);
        z = norminv(1 - conf_level);
        % Parametric VaR (Gaussian)
        VaR_param = -(mu + SD * z);
        Var_Parametric(i, j) = VaR_param;
        % Non-Parametric VaR (Historical Simulation)
        VaR_np = -prctile(iter_data, (1 - conf_level)*100);
        Var_NonParametric(i, j) = VaR_np;
        % Monte Carlo VaR: simulate 500 scenarios with drift adjustment
        k = mu * window - ((SD * sqrt(window))^2)/2;
        sim_returns = normrnd(0,1,500,1) * sqrt(n/250) * (SD * sqrt(window)) + k * n/250;
        VaR_mc = prctile(sim_returns, (1-conf_level)*100);
        Var_MonteCarlo(i, j) = VaR_mc;
    end 
end
toc;

% Plot: VaRs Over Time at 90% Confidence Level
VaR_level_index = 90; % Select index corresponding to 90% confidence
additional_rows = zeros(window, 1);
Var_Parametric_plot = [additional_rows; Var_Parametric(:, VaR_level_index)];
Var_NonParametric_plot = [additional_rows; Var_NonParametric(:, VaR_level_index)];
Var_MonteCarlo_plot = [additional_rows; Var_MonteCarlo(:, VaR_level_index)];

figure('Color',[1 1 1])
plot(portfolio_returns, 'LineWidth', 0.25); hold on;
plot(-Var_Parametric_plot, 'LineWidth', 1.5, 'LineStyle', '-.');
plot(-Var_NonParametric_plot, 'LineWidth', 1.5, 'LineStyle', '--');
plot(Var_MonteCarlo_plot, 'LineWidth', 1, 'LineStyle', ':');
xlabel('Time (Days)','Interpreter','latex')
ylabel('VaR / Portfolio Returns','Interpreter','latex')
legend('Portfolio Returns','Parametric VaR','Non-Parametric VaR','Monte Carlo VaR', 'Interpreter','latex')
title('VaRs Over Time at 90% Confidence','Interpreter','latex')
grid on;
hold off;

% Plot: VaR vs Confidence Level Comparison (for the first iteration)
figure('Color',[1 1 1])
subplot(1,3,1)
plot(confidence_interval, Var_Parametric(1,:), 'LineWidth', 0.5)
title('Parametric VaR vs Confidence','Interpreter','latex')
xlabel('Confidence Level','Interpreter','latex')
ylabel('VaR','Interpreter','latex')
grid on

subplot(1,3,2)
plot(confidence_interval, Var_NonParametric(1,:), 'LineWidth', 0.5)
title('Non-Parametric VaR vs Confidence','Interpreter','latex')
xlabel('Confidence Level','Interpreter','latex')
ylabel('VaR','Interpreter','latex')
grid on

subplot(1,3,3)
plot(confidence_interval, Var_MonteCarlo(1,:), 'LineWidth', 0.5)
title('Monte Carlo VaR vs Confidence','Interpreter','latex')
xlabel('Confidence Level','Interpreter','latex')
ylabel('VaR','Interpreter','latex')
grid on;

%% End of Script
