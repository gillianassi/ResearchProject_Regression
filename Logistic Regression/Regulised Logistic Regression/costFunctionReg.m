function [J, grad] = costFunctionReg(theta, X, Y, lambda)
% Code template taken from Andrew Ng MachineLearning:
% http://openclassroom.stanford.edu/MainFolder/CoursePage.php?course=MachineLearning
% rest of code is written by myself
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
n = length(Y); % number of training examples

J = 0;
grad = zeros(size(theta));
% ====================== My CODE HERE ======================
z = X * theta;
h = sigmoid(z);

% Calculate gradient.
% theta_0
 grad(1) = 1/n*sum((h-Y).*X(:,1));
% theta_1 - theta_n
for i = 2 : numel(theta)    
    grad(i) = 1/n*sum((h-Y).*X(:,i))+lambda/n*theta(i);
end
logL = sum(Y.*log(h)+(1-Y).*log(1-h));
J = -1/n*logL+lambda/(2*n)*sum(theta.^2);
% =============================================================
end

