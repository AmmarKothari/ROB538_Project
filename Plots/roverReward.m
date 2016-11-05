clear all;
close all;
clc;

% File Params
FILENAME  = '../RWD_0';

AV_WINDOW = 100;

file = csvread(FILENAME);

popSize = size(file,2);
TimeLimit = size(file,1)-mod(size(file,1),AV_WINDOW);
file = file(1:TimeLimit,:);

% System reward (G) history
figure;
hold on;
for p = 1:popSize
    aux = file(:,p);
    aux = reshape(aux,AV_WINDOW,TimeLimit/AV_WINDOW);
%         plot(1:AV_WINDOW:TimeLimit,aux);
    errorbar(1:AV_WINDOW:TimeLimit,mean(aux),2*std(aux));
end
xlabel('Generation');
ylabel('Local reward (L)');
grid on;
axis tight;