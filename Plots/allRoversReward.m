clear all;
close all;
clc;

% File Params
FILENAME  = '../RWD_';

AV_WINDOW = 50;
NUM_ROVERS = 3;

% System reward (G) history
figure;
hold on;
for r = 1:NUM_ROVERS
    file = csvread(strcat(FILENAME,num2str(r-1)));
    TimeLimit = size(file,1)-mod(size(file,1),AV_WINDOW);
    aux = file(1:TimeLimit,end);    
    aux = reshape(aux,AV_WINDOW,TimeLimit/AV_WINDOW);
    errorbar(1:AV_WINDOW:TimeLimit,mean(aux),2*std(aux));
end
xlabel('Generation');
ylabel('Local reward (L)');
grid on;
axis tight;