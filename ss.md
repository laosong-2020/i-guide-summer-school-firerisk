Method
The method in this research is using PCA to do some pre-process for the data set, that would make
the next steps become easier. After the pre-process, find out the most related characters that causes

the forest fire, then use linear regression to process the data, to get the first result, and then use random
forest as the second method, to get the second result. Then compare each method and their result, to
see which method would be more accurate in predicting forest fires

**Method**

The method in this research is using PCA to do some pre-process for the data set, that would make
the next steps become easier. After the pre-process, find out the most related characters that causes the forest fire, then use **linear regression** to process the data, to get the first result, then use **random forest** as the second method, to get the second result, lastly, the **gradient boosting** method for the third result. Then compare each method and their result, to see which method would be more accurate in predicting forest fires.

**PCA Data Process**
PCA (Principal Component Analysis) is a widely used data condense method. In this process, it
drops off some irrelevant features, and keep all the important features that have large impact to
the result. And that would help to do the analysis much easier. In this case, finding the character that
have the most impact to the forest fire will become the job to do. So among all the 7 features
in the data set **(Exposure type, Burn-probability,Risk2-Potential-structures, Length-over-8-feet, length-over-4-fee, Conditional-risk2-Potential-Structures and Conditional-flame-length))**, it will find the 2 features that have the most impact on wild fire. So in this process, first we will find the explained variance ratio, that is the number stands for the relativity of the character to the result, the closer the result number to 0, more related the features are. After getting all the explained variance ratio, we will compare them to each other and then the relativity, or the impact of the features to the result can be get.

1. **Linear Regression Model**

What is Linear Regression :
Linear regression is a method to analysis the relationship between one or more independent variable and dependent variable using regression formula. It’s the linear combination of one or more returning model . 

When to use:

Linear regression can be used to make predictions by using data sets to make a prediction model, then if there is a new variable 𝑋 and there is no existing 𝑌 to fit that 𝑋, that model can be used to predict a new 𝑌. Or giving a variable 𝑌 and some variable 𝑋1, … , 𝑋𝑝, all those 𝑋 can have relationship to 𝑌 , and linear regression can be used to evaluate the strength of relationship between 𝑌 and any random 𝑋𝑟 , and recognize which 𝑋 has the unnecessary information about 𝑌.

Pros/Advantages

Simple model : The Linear regression model is the simplest equation using which the relationship between the multiple predictor variables and predicted variable can be expressed.

Computationally efficient : The modeling speed of Linear regression is fast as it does not require complicated calculations and runs predictions fast when the amount of data is large.

Interpretability of the Output: The ability of Linear regression to determine the relative influence of one or more predictor variables to the predicted value when the predictors are independent of each other is one of the key reasons of the popularity of Linear regression. The model derived using this method can express the what change in the predictor variable causes what change in the predicted or target variable.

Cons/Disadvantages

Overly-Simplistic: The Linear regression model is too simplistic to capture real world complexity

Linearity Assumption: Linear regression makes strong assumptions that there is Predictor (independent) and Predicted (dependent) variables are linearly related which may not be the case.

Severely affected by Outliers: Outliers can have a large effect on the output, as the Best Fit Line tries to minimize the MSE for the outlier points as well, resulting in a model that is not able to capture the information in the data.

Independence of variables :Assumes that the predictor variables are not correlated which is rarely true. It is important to, therefore, remove multicollinearity (using dimensionality reduction techniques) because the technique assumes that there is no relationship among independent variables. In cases of high multicollinearity, two features that have high correlation will influence each other’s weight and result in an unreliable model.

Assumes Homoskedacity :Linear regression looks at a relationship between the mean of the predictor/dependent variable and the predicted/independent variables and assumes constant variance around the mean which is unrealistic in most cases.

Inability to determine Feature importance :As discussed in the “Assumes independent variables” point, in cases of high multicollinearity, 2 features that have high correlation will affect each other’s weight. If we run stochastic linear regression multiple times, the result may be different weights each time for these 2 features. So, it’s we cannot really interpret the importance of these features.

Complexity

The basic principle of linear regression is to find a line that could best fit all the data points in a
data set. But since all the data point cannot perfectly fit the line, so there must be error during the
linear regression. 

Formula

In order to find out which line can represent all the data point the best, here’s the
formula of the loss function.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cb0dfb90-df60-4e9d-906b-98dc4a714221/f795559f-0abc-439e-b9cd-62f0f6085ebb/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cb0dfb90-df60-4e9d-906b-98dc4a714221/97144a69-316e-4147-b27e-c95a731154a9/Untitled.png)

**2. Random Forest**

What is Random Forest model:?

Random forest method also has a very wide application prospect. It can be used from marketing
to medical insurance. It is an integrated method and has the base unit called decision tree . 

When to use:

This method integrates more than one decision tree by using ensemble learning. So basically, is to input a simple, and get a result from each decision tree and integrate all of the results from every one of the decision trees. Then find the final result. And because the simple is processed for many times from each decision trees, so the result will have excellent accuracy. And because it doesn’t need variable deletion, it will be able to process high dimensional data sets and will be very efficient. In that case using random forest in forest fire prediction would a properly choice. The basic rule of random forest is to set there are 𝑛 forest fire data sets, and 𝑚 weather causes. Then use bootstrap to draw the random data sets from 𝑛 forest fire data sets, in order to get a 𝑛𝑡 as the model collection of 𝑛, then to get the random forest tree of 𝑛𝑡, next draw 𝑚𝑡𝑒𝑠𝑡 (𝑚𝑡𝑒𝑠𝑡 ≤ 𝑚) weather causes, and choose the one with most classifications to do the subfield, let each tree grow as they can and do not do any cut off. Then all the 𝑛𝑡 trees become the random forest and choose the mode of the 𝑛𝑡 trees to be the result of the random forest. All the data sets not drawn from the 𝑛𝑡 can be formed to be the out-of-bag(OOB) data sets, those can be used as the test sets of the random forest model.
In building up the random forest, 𝑛𝑡and 𝑚𝑡𝑒𝑠𝑡 are the two most important custom parameters.
There are experiments proved that 𝑚𝑡𝑒𝑠𝑡 = √𝑚 would be a better choice, and for 𝑛𝑡, as long as
the 𝑛𝑡 value is large enough, the random forest will become accurate.

Pros/ Advantages:

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cb0dfb90-df60-4e9d-906b-98dc4a714221/e99379c7-015e-4c64-bbba-a9b2f2ce5bc4/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cb0dfb90-df60-4e9d-906b-98dc4a714221/28636f5e-0e1d-4762-b73c-20009e042077/Untitled.png)

Cons / Disadvantages

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cb0dfb90-df60-4e9d-906b-98dc4a714221/e38d331f-ae22-4e0e-b68f-539989ec652b/Untitled.png)

Formula

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cb0dfb90-df60-4e9d-906b-98dc4a714221/261d6084-07ea-4d3a-976e-ca3dd07f4582/Untitled.png)

**Result**
This research will start with how the data processed, and compare the results to discuss and find
out the best way to predict the forest fires.

**Performance and accuracy compare**
In order to compare the performance of the three methods used in this research, there are several
index can be used for the checking and the compare. The first one is Root Mean Squared Error(MSE). is a metric that measures the average magnitude of errors and deviations from an actual value. It's calculated by taking the square root of the mean squared error (MSE), which is **the average of squared errors**. 

 And the second one is R2 score. R2 score is the ratio of the square of the regression and the total square. Normally larger the R2 score is, means the independent variable has a more explain rate to the dependent variable. And smaller the R2 score is, means the method better fits model.

**Reference**

Guan, R. (2023). Predicting forest fire with linear regression and random forest. *Highlights in Science, Engineering and Technology*, *44*, 1-7.

https://medium.com/@satyavishnumolakala/linear-regression-pros-cons-62085314aef0

[https://aiml.com/what-are-the-advantages-and-disadvantages-of-random-forest/#:~:text=Random forest can be computationally,when working with limited resources.&text=Although random forest is resistant,when working with noisy data](https://aiml.com/what-are-the-advantages-and-disadvantages-of-random-forest/#:~:text=Random%20forest%20can%20be%20computationally,when%20working%20with%20limited%20resources.&text=Although%20random%20forest%20is%20resistant,when%20working%20with%20noisy%20data).