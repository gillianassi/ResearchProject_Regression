% Code is provided in this file from 
%http://openclassroom.stanford.edu/MainFolder/CoursePage.php?course=MachineLearning
% the function costFunctionReg was the main issue

%% Plot function
%  The first two columns contains the X values and the third column
%  contains the label (y).
clear all; close all; clc
data = load('RegulisedLogisticRegression.txt');
X = data(:, [1, 2]); Y = data(:, 3);
plotData(X, Y);
% Put some labels 
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;

%%  Add Polynomial Features
% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = mapFeature(X(:,1), X(:,2));

%% Add sigmoid function, cost function and gradient

% Initialize fitting parameters
init_theta = zeros(size(X, 2), 1);
% Set regularization parameter lambda to 1
lambda = 1;
% compute cost and gradient
[cost, grad] = costFunctionReg(init_theta,X,Y,lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);

%% Optimize
% Initialize fitting parameters
init_theta = zeros(size(X, 2), 1);
% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;
% Set provided Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, Y, lambda)), init_theta, options);

%% STEP 7
% Plot Boundary
plotDecisionBoundary(theta, X, Y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;