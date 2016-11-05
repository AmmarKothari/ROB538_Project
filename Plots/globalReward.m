clear all;
close all;
clc;

% File Params
FILENAME = {'../LocalRwd/RWD_G', '../GlobalRwd/RWD_G', '../DiffRwd/RWD_G'};

AV_WINDOW = 10;
NUM_ROVERS = 3;

% System reward (G) history
figure;
hold on;
for r = 1:3
    file = csvread(FILENAME{r});
    TimeLimit = size(file,1)-mod(size(file,1),AV_WINDOW);
%     file = mean(file')';
    file = file(:,end);
    aux = file(1:TimeLimit);
    aux = reshape(aux,AV_WINDOW,TimeLimit/AV_WINDOW);
    errorbar(1:AV_WINDOW:TimeLimit,mean(aux),2*std(aux));
end
xlabel('Generation');
ylabel('Global reward (L)');
grid on;
axis tight;
legend('L','G','D')