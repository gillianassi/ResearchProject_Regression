% Logistic Regression
%% Step 1
clear all; close all; clc
scores = load('TestScores.dat');
admitted = load('Students.dat');
[m, n] = size(scores); 
%% Plot data
% Add intercept term to scores
scores = [ones(m,1),scores];
% Plot the training data
% Use different markers in the plot for positives and negatives
figure
pos = find(admitted==1);
neg = find(admitted==0);
plot(scores(pos, 2), scores(pos,3), '+')
hold on
plot(scores(neg, 2), scores(neg, 3), 'o')
hold on
xlabel('Exam 1 score')
ylabel('Exam 2 score')
%% Implement Newton's function to minimize the cost function
% Initialize theta
theta = zeros(n+1,1);
% Define the sigmoid function
g = @(z) 1.0 ./ (1.0 + exp(-z)); 
% Newton's method
MAX_ITR = 7; % it will converge after 5-15 itterations
J = zeros(MAX_ITR, 1);
for i = 1:MAX_ITR
    % Calculate the hypothesis function
    z = scores * theta;
    h = g(z);
    % Calculate gradient and hessian.
    grad = (1/n)*scores.'*(h-admitted);
    H = (1/n)*scores.'*diag(h)*diag(1-h)*scores;
    % Calculate J (for testing convergence)
    logl = sum(admitted.*log(h)+(1-admitted).*log(1-h)); 
    J(i) = -(1/m)*logl(1);
    % now update theta
    theta = theta- inv(H)*grad;
end
% Display theta
theta
%% Calculate Decision boundry
% Only need 2 points to define a line, so choose two endpoints
plot_scores = [15,65];
% Calculate the decision boundary line
plot_admitted = -(theta(1)+theta(2).*plot_scores)/theta(3);
plot(plot_scores, plot_admitted)
legend('Admitted', 'Not admitted', 'Decision Boundary')
hold off

% Plot J
figure
plot(0:MAX_ITR-1, J, 'o--', 'MarkerFaceColor', 'b', 'MarkerSize', 10)
xlabel('Iteration'); ylabel('J')
% Display J
J
%% Calculate the probability 
% student with Score 20 on exam 1 and score 80 on exam 2 
% will NOT be admitted --> 1- g(z)
z = theta(1)+theta(2).*20+theta(3).*80; 
prob = 1 - g(z)