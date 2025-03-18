%% Q3_VaR_Bond.m
% VAR OF A BOND & 10% Price Decline Probability via YTM Path Simulation
%
% 이 스크립트는 다음을 수행합니다.
% 1) 10년 만기, 연 5% 쿠폰, 액면가 100, 현재 가격 99인 채권의 YTM을 구합니다.
% 2) 매일의 YTM 경로를 시뮬레이션하여, 30일 동안 한 번이라도 채권 가격이 10% 하락(89.1 이하)
%    하는 경우의 확률을 추정합니다.
% 3) 다양한 기간(1, 10, 20, …, 90일)에 대해 6가지 방법(정확한 공식, 델타 근사, 델타-감마 근사,
%    MC 델타, MC 델타-감마, MC 전체 재평가)으로 VaR를 계산합니다.
%
% 참고: 채권 가격은 YTM에 대해 분수함수의 형태를 가지므로, YTM 상승 시 가격이 하락합니다.
%       따라서 옵션 가격 산출 시, 손실(가격하락)을 올바르게 반영하기 위해
%       정확한 공식에서는 YTM 충격에 절대값(|z|)을 사용해야 합니다.
%
% 가정:
% - 액면가 = 100, 쿠폰율 = 5%, 만기 = 10년, 현재 가격 = 99.
% - 1년 = 360일.
% - 일일 YTM 변동 ~ N(0, 0.006^2) (i.i.d.)
% - Monte Carlo 시뮬레이션은 최소 10,000번 사용

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
% z-score for 1% quantile (하방 움직임: 원래 음수이나, 손실 계산에는 절대값 사용)
z_0_01 = -2.326;

%% Calculate Yield to Maturity (YTM)
% 방정식: annual_coupon*sum_{t=1}^{10}(1/(1+y)^t) + FV/(1+y)^10 = 99
syms x positive
eqn = annual_coupon*sum(1./(1+x).^(1:maturity)) + FV/(1+x)^maturity == P0;
sol = vpasolve(eqn, x, [0,1]);
YTM = double(sol);
fprintf('Yield to Maturity (annual): %.4f\n', YTM);

%% Compute Duration and Convexity (Analytical Approximations)
t = (1:maturity)';
price_calc = annual_coupon*sum(1./(1+YTM).^(t)) + FV/(1+YTM)^maturity;
duration = (annual_coupon*sum(t./(1+YTM).^(t)) + maturity*FV/(1+YTM)^maturity) / price_calc;
convexity = (annual_coupon*sum(t.*(t+1)./(1+YTM).^(t+2)) + maturity*(maturity+1)*FV/(1+YTM)^(maturity+2)) / P0;
fprintf('Duration: %.4f\n', duration);
fprintf('Convexity: %.4f\n', convexity);

% 수정 듀레이션 (Modified Duration)
mod_duration = duration/(1+YTM);

%% Define bondPrice function
% h일 경과 후 남은 만기: maturity - h/360 (연 단위)
% 남은 쿠폰은 h/360 이후에 지급되는 것들만 포함
bondPrice = @(y, h) ( ...
    sum( annual_coupon ./ (1+y).^( max((1:maturity)-h/360,0) ) .* ( (1:maturity) > h/360 ) ) ...
    + FV/(1+y)^(maturity - h/360) );

%% 1) 10% Price Decline Probability within 30 Days via YTM Path Simulation
% 각 경로마다 30일 동안 매일의 YTM 변동을 생성하여 누적한 후,
% 매일 채권 가격을 계산하고, 한 번이라도 가격이 89.1 이하로 내려간 경우를 카운트.
h_decline = 30;
num_sim_decline = 10000;
threshold = 0.9 * P0;  % 89.1

count_decline = 0;
for i = 1:num_sim_decline
    % 30일간 매일의 YTM 변동 생성
    dy_daily = normrnd(daily_mean, daily_std, h_decline, 1);
    % 누적 YTM 경로: 초기 YTM에 매일 누적합
    ytm_path = YTM + cumsum(dy_daily);
    hit = false;
    % 각 일자별로 채권 가격 계산 (남은 기간 = maturity - (d/360))
    for d = 1:h_decline
        price_d = bondPrice(ytm_path(d), d);
        if price_d <= threshold
            hit = true;
            break;
        end
    end
    if hit
        count_decline = count_decline + 1;
    end
end
prob_decline_path = count_decline / num_sim_decline;
fprintf('Probability of 10%% decline within 30 days (path simulation): %.4f\n', prob_decline_path);

%% 2) VaR Calculations using Exact, Delta, Delta-Gamma formulas
num_horizons = length(horizons);
VaR_exact = zeros(num_horizons,1);
VaR_delta = zeros(num_horizons,1);
VaR_delta_gamma = zeros(num_horizons,1);

% 델타 (수정 듀레이션 이용): dP/dy ≈ -P0*Duration/(1+YTM)
dPdy = -P0 * duration/(1+YTM);
% 감마 (컨벡시티 이용): d²P/dy² ≈ P0 * convexity
d2Pdy2 = P0 * convexity;

for i = 1:num_horizons
    h = horizons(i);
    % shock: 1% 분위수에 해당하는 YTM 변화 (절대값 사용)
    dy = abs(z_0_01) * daily_std * sqrt(h);
    
    % 1) Exact formula: YTM 상승 (절대값 사용)로 채권 가격 재평가
    newYTM_exact = YTM + dy;
    P_h_exact = bondPrice(newYTM_exact, h);
    VaR_exact(i) = P0 - P_h_exact;  % 손실 (양수여야 함)
    
    % 2) Delta approximation: ΔP ≈ -dP/dy * Δy
    VaR_delta(i) = -dPdy * dy;
    
    % 3) Delta-Gamma approximation: ΔP ≈ -dP/dy*dy - 0.5*d2P/dy2*dy^2
    VaR_delta_gamma(i) = -dPdy * dy - 0.5 * d2Pdy2 * dy^2;
    
    % 조건 검증: 기대컨디션은 Delta Approx > Exact > Delta-Gamma Approx 이어야 함.
    if ~( VaR_delta(i) > VaR_exact(i) && VaR_exact(i) > VaR_delta_gamma(i) )
        warning('At horizon %d days: Ordering condition violated: Delta = %.4f, Exact = %.4f, Delta-Gamma = %.4f', ...
            h, VaR_delta(i), VaR_exact(i), VaR_delta_gamma(i));
    end
end

%% Monte Carlo Simulations for VaR (최소 10,000번)
num_sim = 10000;
VaR_delta_MC = zeros(num_horizons,1);
VaR_delta_gamma_MC = zeros(num_horizons,1);
VaR_full_MC = zeros(num_horizons,1);

for i = 1:num_horizons
    h = horizons(i);
    % h일 동안의 YTM 변동 (한 번의 샘플은 h일 누적 변화)
    dy_sim = normrnd(daily_mean, daily_std*sqrt(h), num_sim, 1);
    
    % 4) MC: Delta approximation
    P_sim_delta = P0 + dPdy * dy_sim;
    loss_delta = P0 - P_sim_delta;
    VaR_delta_MC(i) = quantile(loss_delta, conf_level);
    
    % 5) MC: Delta-Gamma approximation
    P_sim_dg = P0 + dPdy * dy_sim + 0.5 * d2Pdy2 * dy_sim.^2;
    loss_dg = P0 - P_sim_dg;
    VaR_delta_gamma_MC(i) = quantile(loss_dg, conf_level);
    
    % 6) MC: Full revaluation - 각 시뮬레이션 경로에서 단일 YTM 변화로 재평가
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
% VaR 추정치 시각화
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

% Expected Shortfall 시각화
figure;
plot(horizons, ES_full, '-o', 'LineWidth', 1.5);
xlabel('Horizon (days)');
ylabel('Expected Shortfall');
title('Expected Shortfall (Full Revaluation MC) vs. Horizon');
grid on;
saveas(gcf, 'ExpectedShortfall.png');

%% Statistical Test
% 예시: MC Full 재평가 VaR와 MC Delta-Gamma VaR 간 차이를 t-test로 비교
[h_test, p_value] = ttest(VaR_full_MC, VaR_delta_gamma_MC);
fprintf('T-test (MC Full vs. MC Delta-Gamma): h = %d, p-value = %.4f\n', h_test, p_value);
