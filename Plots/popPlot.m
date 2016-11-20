clear all;
close all;
clc;

% File Params
FILENAME = {'../LocalRwd/SYS_RWD', '../GlobalRwd/SYS_RWD', '../DiffRwd/SYS_RWD'};

AV_WINDOW = 10;
NUM_ROVERS = 2;

figure;
hold on;
file = csvread(FILENAME{2});
plot(max(file'));
plot(mean(file'));
plot(min(file'));
xlabel('Generation');
ylabel('System reward');
grid on;
axis tight;
legend('max','mean','min')