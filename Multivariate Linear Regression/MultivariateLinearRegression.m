clear all; close all; clc
%% Step 1: Loading data
houses = load('LivingArea_Bedrooms.dat') ;
prices = load('Prices.dat');
n = length(prices);
% Add intercept term to x
houses = [ones(n,1),houses];
% Save a copy of the unscaled features for later
houses_unscaled = houses;
%% Step 2: Preprocessing
% Scale features and set them to zero mean
mu = mean(houses);
sigma = std(houses);
houses(:,2) = (houses(:,2) - mu(2)) / sigma(2);
houses(:,3) = (houses(:,3) - mu(3)) / sigma(3);
% Prepare for plotting
figure;
% plot each alpha's data points in a different style braces indicate a cell
% not just a regular array.(x_unscaled' * x_unscaled)^-1 * x_unscaled' * y
plotstyle = {'b', 'r', 'g', 'k', 'b--', 'r--'};
%% Step 3: Prameter initialization
% In this step, you will generate random parameters for theta
rng(10);
theta_grad_descent = rand(1,3);
%% Step 4: Gradient Descent 
alpha = [0.01, 0.03, 0.1, 0.3, 1, 1.3]; % Different learning rates
MAX_ITR = 100;
for i = 1:length(alpha)
    theta = zeros(size(houses(1,:)))'; % initialize fitting parameters
    J = zeros(MAX_ITR, 1); %initialize cost function
    for num_iterations = 1:MAX_ITR
        % Calculate the J 
        J(num_iterations) =1/(2*n)*(houses*theta-prices).'*(houses*theta-prices);
        % The gradient
        grad = 1/n*houses.'*(houses*theta-prices);
        % Update theta
        theta = theta-alpha(i)*grad;
    end
    % Now plot the first 50 J terms
    plot(0:49, J(1:50), char(plotstyle(i)), 'LineWidth', 2)
    hold on
    theta_grad_descent(i,:) = theta;
end
legend('0.01','0.03','0.1', '0.3', '1', '1.3')
xlabel('Number of iterations')
ylabel('Cost J')
% force Matlab to display more than 4 decimal places
% formatting persists for rest of this session
format long
% Display gradient descent's result 
best_theta = theta_grad_descent(5,:) % best theta with fastest convergance
% Estimate the price of a 1650 sq-ft, 3 br house
price_grad_desc = dot(best_theta, [1, (1650 - mu(2))/sigma(2),...
                    (3 - mu(3))/sigma(3)]);
price_grad_desc
%% Step 5: Normal Equation
% Calculate the parameters from the normal equation
theta_normal = inv(houses_unscaled.'*houses_unscaled)*(houses_unscaled.'*prices)

%Estimate the house price again
price_normal = dot(theta_normal, [1, 1650, 3]);
fprintf("predicted price from normal equation: ");price_normal
%% Step 6: Regularization
lambda = 0.01; % you can try with different value of lambda
theta_normal_regularized = inv(houses_unscaled.'*houses_unscaled+lambda*eye(3))*houses_unscaled.'*prices;
price_normal_regularized = dot(theta_normal_regularized, [1, 1650, 3])