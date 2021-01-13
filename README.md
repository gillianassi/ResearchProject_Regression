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
Linear regressiion is a frequently used method to analyze an available data set. The most commonly used form of linear regression is known as **least squares fitting**. This form aims to fit a polynomial curve to data, in a way that the sum of squares of the distance from the line to the data points is minimized.
##### Simple Linear Regression
##### Gradient Descent

#### Multivariate Linear Regression

#### Logistic Regression
##### Newton’s Method
##### Regulised Logistic Regression


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
