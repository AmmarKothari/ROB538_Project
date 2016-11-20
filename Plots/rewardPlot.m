clear all;
close all;
clc;

% File Params
FILENAME = {'../LocalRwd/SYS_RWD', '../GlobalRwd/SYS_RWD', '../DiffRwd/SYS_RWD'};
LEGEND = {'L','G','D'};
RANGE = 1:3;    
% RANGE = 1;

figure; 
hold on;
for r = RANGE
    file = csvread(FILENAME{r});    
    plot(mean(file'));
end
xlabel('Generation');
ylabel('System reward');
grid on;
axis tight;
legend(LEGEND{RANGE})