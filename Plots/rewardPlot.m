clear all;
close all;
clc;

% File Params
FILENAME = {'../LocalRwd/SYS_RWD', '../GlobalRwd/SYS_RWD', '../DiffRwd/SYS_RWD'};

figure;
hold on;
for r = 1:3
    file = csvread(FILENAME{r});    
    plot(mean(file'));
end
xlabel('Generation');
ylabel('System reward');
grid on;
axis tight;
legend('L','G','D')