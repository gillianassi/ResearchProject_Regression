# Research Project Gameplay Programming
## Regression
### Goal

The purpose of this research is to broaden my knowledge about artificial intelligence (AI), more specifically, **regression**. This will be done by implementing a set of exercises with both self-generated data and existing datasets to get a better grasp of the subject. Both advantages and drawbacks will be discussed for each type of regression analysed during this research.

### Regression
Modern AI's are machine learning solutions created for a **specific task**. This can vary from understanding and analysing trends in market shares to finding the optimal location for a pizza shop. These AI's aren't considered *real intelligence* but use statistics and analytics to handle different problems dependant on their given training.
The way that such AI's learn can be subdivided into three categories:
* *Supervised Learning*, in which the algorithms are trained for a specific task with pre-existing datasets and human supervision.
* *Unsupervised learning*, where the algorithm looks for previously undetected patterns in pre-existing datasets with as less human supervision as possible.
* *Reinforcement Learning*, in which the algorithms learn using trial and error by adding positive and negative values to experiences and situations.

AI's that combine different aspects of learning also exist but are out of the scope of this research.

#### Supervised Learning
The most relevant training method for this research is *Supervised Learning*. During supervised learning the goal is to approximate a mapping function *f* from input *x* to output *Y* using algorithms. 
<p align="center">
Y = f(x) 
</p>

This method is referred to as supervised learning because the creator knows what the correct answers need to look like. While the algorithm makes predictions on the training data, the creator will correct if it is necessary. When a certain level of accuracy is achieved, the learning stops. Supervised learning problems can be subdivided into **classification** and **regression** problems. The output of both problems will, as their name suggests, be different. Classification problems will output a *category* and tries to classify the received datasets. Regression problems, on the other hand, predict a *numeric value in any range* as output variable, such as "height" or "speed".

This research focusses on the different types of regression algorithms, who try to predict dependant values based on known datasets. This is done by looking at the relationship between different values. The following section will explain some of the most commonly used regression algorithms followed by an implementation explaining its use.

#### Linear Regression
Linear regressiion is a frequently used method to analyze an available data set. 

##### Simple Linear Regression
###### General explenation
The most commonly used form of linear regression is known as **least squares fitting**. This form aims to fit a polynomial curve to data, in a way that the sum of squares of the distance from the line to the data points is minimized.

When the least squeres estimator of a linear regression model with a single explanatory variable is called **simple linear regression**. This fits a straight line trough a set of n points in such a way that makes the sum of squared residuals as small as possible. A visualisation of this process can be seen in the following image:

<p align="center"><img src="Images/LeastSquaresFitting.png" alt="LSF" width="250"/></p>

Suppose there are n training samples  <ins>x</ins><sub>i</sub>= (1, x<sub>i1</sub>)<sup>T</sup> and y<sub>i</sub>, where i = 1, 2, ... , n. 
These samples represent the input random vector <br/> <ins>X</ins> = (1, X<sub>i</sub>)<sup>T</sup>and the output random variable Y, respectively.
The following function describes <ins>x</ins> <sub>i</sub> and y<sub>i</sub>:

<p align="center">
y<sub>i</sub> = <ins>θ</ins><sup>T</sup> <ins>x</ins><sub>i</sub> + ϵ<sub>i</sub> = θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + ϵ<sub>i</sub>,
</p>

where <ins>θ</ins> = (θ<sub>0</sub>, θ<sub>1</sub>)<sup>T</sup> can be seen as the vector of the parameters and ϵ<sub>i</sub> is the error for a pair
(x<sub>i</sub>, y<sub>i</sub>). The objective is to find the equation of the straight line:

<p align="center">
Y = θ<sub>0</sub> + θ<sub>1</sub>X<sub>1</sub>
</p>

This provides the most optimal fit for the data points, where the sum of squared residuals are minimized. This would mean that the y-intercept θ<sub>0</sub> and the slope θ<sub>1</sub> solve the following minimization problem:

<p align="center"><img src="Images/MinimizationProblem.png" alt="Minimization" width="300"/></p>

By expanding to get a quadratic expression in θ<sub>0</sub> and θ<sub>1</sub>, it can be shown that the minimizing values of θ<sub>0</sub> and θ<sub>1</sub> of the formula above are:

<p align="center"><img src="Images/ThetaValues.png" alt="values" width="150"/></p>

where x<sub>1</sub> = (x<sub>11</sub>, ... , x<sub>i1</sub>, ... , x<sub>n1</sub>) and y = (y<sub>11</sub>, ... , y<sub>i1</sub>, ... , y<sub>n1</sub>) are the row
vectors that contain all sample values of the variables X<sub>1</sub> and Y, respectively. Note that the circumflex over a quantity indicates the sample average. For a more detailed description of the simple linear regression theory, I would recommend the following sources:
* Section 3.1, Chapter 3 from the book “Pattern recognition and machine learning” of Bishop and Nasrabadi.
* https://mathworld.wolfram.com/LeastSquaresFitting.html

###### Implementation: Introduction
>The files of this implementation can be found [here](Simple%20Linear%20Regression/01_Introduction).

To understand the concept of simple linear regression I generated some experimental data adding artificial noise using the equation <br/> y = a<sub>0</sub> + a<sub>1</sub> * x, where a<sub>0</sub> = 2 and a<sub>1</sub> = 1. This is shown in the following image, containing a yellow line indicating the computed linear regression:

<p align="center"><img src="Images/Example1SLR.png" alt="SLR1" width="500"/></p>

The regression line is calculated using built-in Matlab functions to quickly get the grasp of the concept. <br/>
By playing with the standard deviation of the errors, it is noticeable  that the calculated regression becomes less accurate.

<p align="center"><img src="Images/Example2SLR.png" alt="SLR2" width="500"/></p>

To simulate the effects of a random experimental error, I've repeated this process 1000 times with a fixed standard deviation of 0,1. By analysing the means in a histogram using a Gaussian curve, as shown in the image below, we notice that a<sub>0</sub> is most likely equal to 1, and a<sub>1</sub> = 2. Knowing that our basic function was y = 1 + 2x, we can assume that our approach was effective.

<p align="center"><img src="Images/Example3SLR.png" alt="SLR3" width="500"/></p>

##### Gradient Descent
###### General explenation
Gradient Descent is theoretically an algorithm that minimizes functions. This perfectly fits in the context of regression, where one tries to minimalize the sum of squared residuals. Any function can be defined by a set of parameters <ins>0</ins>. Gradient descent will initialize such a set and gradually move towards a set of parameters to minimize a cost function using calculus.

For each sample <ins>x</ins><sub>i</sub> = (1, x<sub>i1</sub>)<sup>T</sup> the hypothesis function can be defined as


<p align="center">h(<ins>x</ins><sub>i</sub>) = <ins>θ</ins><sup>T</sup> <ins>x</ins><sub>i</sub> = θ<sub>0</sub> + θ<sub>1</sub>x<sub>i1</sub>,</p>

or,
<p align="center">h(<ins>x</ins><sub>i</sub>) = <ins>x</ins><sub>i</sub><sup>T</sup><ins>θ</ins> = θ<sub>0</sub> + θ<sub>1</sub>x<sub>i1</sub>,</p>

The next step is to figure out the parameters θ = (θ<sub>0</sub>, θ<sub>1</sub>)<sup>T</sup> , which will minimize the square error between the predicted value h(<ins>x</ins>) and the actual output y for all values i in the training set. The cost function can then be noted as followed:

<p align="center"><img src="Images/InitialCostFunction.png" alt="InitCost" width="300"/></p>

In this formula, n represents the number of training sets. The scaling of 1/2n is simply notational convenience. This can also be rewritten using matrix notations as

<p align="center"><img src="Images/InitialCostFunctionMatrixNot.png" alt="MatrixCost" width="300"/></p>

To reduce the cost function, θ will need to be updated each iteration  using the following update rule

<p align="center"><img src="Images/UpdateRule.png" alt="UpdateRule" width="500"/></p>

###### Implementation: Gradient Descent
>The files of this implementation can be found [here](Simple%20Linear%20Regression/02_GradientDescent).

For this implementation, I replicated an exercise provided by the learning platform 'openclassroom' in the course *Machine Learning* found [here](http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html). This was done by generating my own relevant data and implementing linear regression. The generated data involves children between the ages of 2 and 8 years old and their heights.

The implementation focusses on using the *update rule*  to update θ. In the following image, the red line indicates the first calculated regression line after 1 iteration of the update rule. This is clearly an inaccurate representation of the dataset, but after 1478 iterations, both theta's have convoluted to a fixed value. The regression line found with these theta values is shown by the yellow line.

<p align="center"><img src="Images/Example1GDSLR.png" alt="SLR3" width="500"/></p>

This regression line can further be used to estimate the average height of children with a specific age, an example can be found in the code.

*Note that it is important to adjust the learning rate with different datasets. This can either make or break the linear regression.*

###### Implementation: Visualisation of Gradient Descent
>The files of this implementation can be found [here](Simple%20Linear%20Regression/02_GradientDescent).

By implementing the definition of the cost function, all possible theta's can be calculated and visualised inside of a surface plot. If this is done using the previously generated dataset the following can be created:

<p align="center"><img src="Images/Example2GDSLR.png" alt="SLR3" width="500"/></p>

The red line indicates the theta's calculated in the previous example. <br/>
*It is important to note that this red line can be found in the valley of the surface plot, where the cost function is at its lowest.*

#### Multivariate Linear Regression
###### General Explenation
It is also possible to analyse the degree of a linear relation of multiple predictors and responses. This is based on the same concept of the previous linear regression but adds extra variables to the field. A detailed description of multivariate regression problems can be found via the following sources:
* Section 2.3.1, Ch. 2 from the book “The Elements of Statistical Learning” of Hastie, Tibshirani and Friedman.
* Section 3.1, Ch. 3 from the book “Pattern recognition and machine learning” of Bishop and Nasrabadi.

###### Implementation: Multivariate Regression
>The files of this implementation can be found [here](Multivariate%20Linear%20Regression).

This implementation is my method used while following an exercise provided by the learning platform 'openclassroom' in the course *Machine Learning* found [here](http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex3/ex3.html). Because the implementation of multivariate linear regression is very similar to simple linear regression, it focusses a bit more at the influence of the learning rate, which can heavily influence the results, as mentioned at the end of **Implementation: Gradient Descent**.

The following graph shows the influence of the learning rates, for the values 0.01, 0.03, 0.1, 0.3, 1, and 1.3.

<p align="center"><img src="Images/Example1MLR.png" alt="SLR3" width="500"/></p>

By analysing the graph, we can assume that a small learning rate will cause the cost function to convolute too slowly. This increases the amount of itteration needed to reach the convoluted value of the cost function. However, This clearly does not mean you can simply increase the learning rate without concequences. By comparing the blue-striped line with the red-striped line, we can see how a higher learningrate can make the convergence slower or even impossible.

Just like in the previous implementation **Implementation: Gradient Descent**, we can use the generated theta's to predict values. However, Normal equations with regularization are necessary to mitigate the chances of the model "overfitting" the training data, which could happen if we allowed  parameters to grow arbitrarily large.
To regulate a linear regression model, a penalty term is used on the square values. The influence of this penalty term is controlled by the parameter λ. <br/>

With y being the target values, X the input features, and λ the regularization parameter, we can find the close form solution for the regularized linear regression with the  following formula:

<p align="center"><ins>θ</ins>= [X<sup>T</sup>X + λ * I]<sup>-1</sup> X<sup>T</sup>y</p>

Just like un-regulized linear regression, the predicted values y is calculated by y = X<ins>θ</ins>.
My implementation of the normalization and regularization can be found at the end of the implementation.


#### Logistic Regression
###### General Explenation

###### Implementation: Newton’s Method


###### Implementation: Regulised Logistic Regression

### Future work

### Conclusion

### Sources
* Machine learning Chapters used (http://openclassroom.stanford.edu/MainFolder/CoursePage.php?course=MachineLearning):
  * http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html
  * http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex3/ex3.html
  * http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex4/ex4.html 
  * http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex5/ex5.html

* Artificial intelligence:
  * https://www.unemyr.com/understanding-ai-algorithms/
* Regression
  * https://www.investopedia.com/terms/r/regression.asp#:~:text=Regression%20is%20a%20statistical%20method,(known%20as%20independent%20variables)
  * https://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/
  * https://www.unemyr.com/ai-algorithms-regression/
* Simple Linear regression
 * http://home.iitk.ac.in/~shalab/econometrics/Chapter2-Econometrics-SimpleLinearRegressionAnalysis.pdf
* Gradient descent
 * https://medium.com/@lachlanmiller_52885/machine-learning-week-1-cost-function-gradient-descent-and-univariate-linear-regression-8f5fe69815fd
* Multivariate Regression
 * https://brilliant.org/wiki/multivariate-regression/#:~:text=Multivariate%20Regression%20is%20a%20method,responses)%2C%20are%20linearly%20related.
 
