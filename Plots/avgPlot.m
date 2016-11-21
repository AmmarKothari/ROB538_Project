clear all;
close all;
clc;

% File Params
FILENAME        = {'../LocalRwd/SYS_RWD', '../GlobalRwd/SYS_RWD', '../DiffRwd/SYS_RWD'};
LEGEND          = {'L','G','D'};
RWD_RANGE       = 1:3;
NUM_TRIALS      = 10;
ERR_BAR_RATIO	= 10;

figure; 
hold on;
for r = RWD_RANGE
    for i = 1:NUM_TRIALS
        file = csvread(strcat(FILENAME{r},int2str(i)));
        if i == 1
            aux = mean(file')';
        else
            aux = [aux mean(file')'];
        end
    end
    m = mean(aux');
    e = std(aux')/sqrt(NUM_TRIALS);
    e3 = repmat(NaN,size(e));
    e3(1:ERR_BAR_RATIO:end) = e(1:ERR_BAR_RATIO:end);
    errorbar(m,e3)
end
xlabel('Generation');
ylabel('System reward');
grid on;
axis tight;
legend(LEGEND{RWD_RANGE})