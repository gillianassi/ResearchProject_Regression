clc;clear;close all;
% Fetch the generated data
age =load('Ages.dat'); % random sorted ages between 2 and 8
height =load('Heights.dat'); % random heights between 0.7 m and 1.3 m
n = size(age,1);
%% Intercept point
IP = ones(n,1);
age = [IP age];
%% initialize J values to 100x100 matrix of zeros
J_val = zeros(100, 100); 
theta0_val = linspace(-3, 3, 100);
theta1_val = linspace(-1, 1, 100);
%% Implement gradient descent to get all possible theta's
for i = 1:length(theta0_val)
    for j = 1:length(theta1_val)
        t = [theta0_val(i); theta1_val(j)];
        h = age*t;
        J_val(i,j) = 1/(2*n)*sum((h-height).^2); % all possible theta's
    end
end
%% Plot the surface plot
% Transpose J to flip the axis for the inbuilt surface plot function
J_val = J_val';
figure
surf(theta0_val, theta1_val, J_val)
xlabel('\theta 0')
ylabel('\theta 1')
zlabel('Cost function')
hold on
line([0.5267,0.5267],[0.0980,0.0980],[0, 45],'Color','red')