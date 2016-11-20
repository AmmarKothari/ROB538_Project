clear all;
close all;
clc;

% File Params
TITLE = {'Local (L)', 'Global (G)', 'Difference (D)'};
FILENAME = {'../LocalRwd/SYS_RWD', '../GlobalRwd/SYS_RWD', '../DiffRwd/SYS_RWD'};

AV_WINDOW = 10;
NUM_ROVERS = 2;

figure;

for i=1:3
    subplot(3,1,i);
    hold on;
    file = csvread(FILENAME{i});
    plot(max(file'));
    plot(median(file'));
    plot(min(file'));
    xlabel('Generation');
    ylabel('System reward');
    grid on;
    axis tight;
    legend('max','median','min');
    title(TITLE{i});
end