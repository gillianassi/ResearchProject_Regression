clc;clear;close all;
% 1. Generate and plot Data of a straight line  ("Correct" line)
%-------------------------------------------------------------
a0 = 2; a1 = 1; 
step = 21; x = linspace(0,1,step);
y1 = a0 + a1 * x; % Equation
plot(x,y1)
hold on

% 2. Generate random errors
%-------------------------------------------------------------
mu = 0; % Zero mean 
sig = 0.1; % standard deviation
Error = normrnd(mu,sig,[1,step]); % 21 random errors
y2 = y1 + Error;
plot(x,y2, 'x','HandleVisibility','off')

% 3. Using a built in function to perform linear regression -->  polyfit 
%-------------------------------------------------------------
p = polyfit(x,y2,1); % get calculated a0 and a1
y3 = polyval(p,x);% get y values with x
plot(x,y3)
% 4. Repeat with different standard deviations
%-------------------------------------------------------------
for sig = 0.2:0.2:1
    % (2. Generate random errors)
    Error = normrnd(0,sig,[1,21]);
    sig
    y2 = y1 + Error;
    plot(x,y2, 'x','HandleVisibility','off')
    % (3. Perform Linear Regression)
    p = polyfit(x,y2,1);
    y3 = polyval(p,x);
    plot(x,y3)
end
legend('original','stdev = 0.1','stdev = 0.2','stdev = 0.4',...
'stdev = 0.6','stdev = 0.8','stdev = 1')

% 5. Simulate the effects of a random experimental error
%    Look at statistical properties of the found regression coëfficients
%    For a standard deviation of 0.1
figure
hold on
a0 = zeros(1,1000); a1 = zeros(1,1000);
for i=1:1001
    % (2. Generate random errors)
    Error = normrnd(0,0.1,[1,21]);
    y2 = y1 + Error;
    plot(x,y2, 'x','HandleVisibility','off')
    % (3. Perform Linear Regression)
    p = polyfit(x,y2,1);
    y3 = polyval(p,x);
    plot(x,y3)
    a0(i) = p(1);
    a1(i) = p(2);
end
figure

% Analyse a0
a0mean = mean(a0)
a0std = std(a0)
subplot(211)
histogram(a0, 'BinWidth', 0.01, 'Normalization', 'pdf')
xgauss = linspace(a0mean-4*a0std,a0mean+4*a0std,1000);
ygauss = normpdf(xgauss,1,a0std);
hold on
plot(xgauss, ygauss, 'r')
title('mean a0') 


% Analyse a1
a1mean = mean(a1)
a1std = std(a1)
subplot(212)
histogram(a1, 'BinWidth', 0.01, 'Normalization', 'pdf')
xgauss = linspace(a1mean-4*a1std,a1mean+4*a1std,1000);
ygauss = normpdf(xgauss,2,a1std);
hold on
plot(xgauss, ygauss,'r')
title('mean a1') 