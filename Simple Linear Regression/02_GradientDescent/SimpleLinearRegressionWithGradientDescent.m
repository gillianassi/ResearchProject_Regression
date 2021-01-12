clc;clear;close all;
% Fetch the generated data
age =load('Ages.dat'); % random sorted ages between 2 and 8
height =load('Heights.dat'); % random heights between 0.7 m and 1.3 m
n = size(age,1);
plot(age,height,'x')
title('Children Height')
xlabel('age (year)') 
ylabel('height (meter)')
hold on
% 3. Intercept point 1
IP = ones(n,1);
age = [IP age];
% 4. Implementation Gradient Descent
alfa = 0.06; % Learning rate adjust with different data sets
alltheta = [0,0];
h = age*alltheta(1,:)';
alltheta(2,1) = alltheta(1,1) - alfa/n*sum((h-height).*age(:,1));
alltheta(2,2) = alltheta(1,2) - alfa/n*sum((h-height).*age(:,2));
plot(age(:,2), age*alltheta(2,:)', '-')
% 5. Iterations
maxir = 1500;
% start from the third itteration, you already have it 1 
for it = 2:maxir+2
    alltheta(it,1) = alltheta(it-1,1) - alfa/n*sum((h-height).*age(:,1));
    alltheta(it,2) = alltheta(it-1,2) - alfa/n*sum((h-height).*age(:,2));
    h = age*alltheta(it,:)';
    if((abs(alltheta(it-1,1)-alltheta(it,1)) < 0.0000001) && ...
        (abs(alltheta(it-1,2)-alltheta(it,2))) < 0.0000001)
        break;
    end
    %plot(age(:,2), age*alltheta(it,:)', '-')% To see gradual progress 
end
it
theta = alltheta(it,:)
plot(age(:,2), age*theta', '-')
legend( 'Linear regression','First itteration','Training data')
ylim([0,3])

% 6. Predictions
child3y50 = 3.5 * theta(2) + theta(1)
child7y00 = 7 * theta(2) + theta(1)