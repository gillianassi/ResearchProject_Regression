clc;clear;
% ages between 2 and 8
data = 2 + (8-2) * rand(50,1) ;    % some random data 
data = sort(data);
save 'Ages.dat' data -ascii   % save to myfile.dat 
load Ages.dat   % load the file

%hights between 0.7 and 1.4

% 1. Generate Data of a straight line ("Correct" line)
% Child of  2 -> 0.7 m to child 8 --> 1.3 m
%-------------------------------------------------------------
a1 = (1.3-0.7)/(8-2); 
a0 = -a1*2 + 0.7;
step = 50; x = linspace(2,8,step);
y1 = a0 + a1 * x; % Equation
% 2. Generate random errors
%-------------------------------------------------------------
mu = 0; % Zero mean 
sig = 0.05; % standard deviation
Error = normrnd(mu,sig,[1,step]); % 21 random errors
data2 =y1 + Error;
data2 = data2';
save 'Heights.dat' data2 -ascii   % save to myfile.dat 
load Heights.dat   % load the file