#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression
# 
# ## Problem Statement -
# 
# In this project, we aim to build a linear regression model to predict sales based on advertising spending across different media channels, namely TV, radio, and newspaper. The dataset provides information on advertising budgets in each channel and the resulting sales figures. Our goal is to identify which advertising medium (TV, radio, or newspaper) has the strongest correlation with sales and can be used as the primary predictor variable for sales prediction.

# ## Step 1: Reading and Understanding the Data
# Let's start with the following steps:
# 
# 1. Importing data using the pandas library
# 2. Understanding the structure of the data

# In[1]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Import numpy and pandas
import numpy as np
import pandas as pd


# In[3]:


# Read the CSV file and view some sample records
advertising= pd.read_csv(r"C:\Users\DELL\Documents\PROJECTS\Simple_Linear_Regression\advertising.csv")
advertising.head()


#  Let's inspect the various aspects of our dataframe

# In[4]:


advertising.shape


# In[5]:


advertising.info()


# In[7]:


advertising.describe()


# ## Step 2: Visualising the Data
# 
# Let's now visualise our data using seaborn. We'll first make a pairplot of all the variables present to visualise which variables are most correlated to `Sales`.

# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:


sns.pairplot(advertising, x_vars=['TV','Radio','Newspaper'], y_vars=['Sales'], size=4 , aspect=1, kind= 'scatter')
plt.show()


# In[16]:


sns.heatmap(advertising.corr(),cmap='YlGnBu',annot=True)
plt.show()


# As is visible from the pairplot and the heatmap, the variable `TV` seems to be most correlated with `Sales`. So let's go ahead and perform simple linear regression using `TV` as our feature variable.

# ---
# ## Step 3: Performing Simple Linear Regression
# 
# Equation of linear regression<br>
# y = mx + c
# 
# -  y is the response
# -  c is the intercept
# -  m is the coefficient for the feature variable
# 
# In our case:
# 
# y = c + m * TV
# 
# The m values are called the model **coefficients** or **model parameters**.
# 
# ---

# ### Generic Steps in model building using `statsmodels`
# 
# We first assign the feature variable, `TV`, in this case, to the variable `X` and the response variable, `Sales`, to the variable `y`.

# In[17]:


x = advertising['TV']
y = advertising['Sales']


# #### Train-Test Split
# 
# Now we need to split our variable into training and testing sets. We'll perform this by importing `train_test_split` from the `sklearn.model_selection` library. It is usually a good to keep 70% of the data in train dataset and the rest 30% in test dataset

# In[18]:


from sklearn.model_selection import train_test_split


# In[39]:


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7, test_size=0.3, random_state=100)


# In[21]:


# Let's now take a look at the train dataset
x_train.head()


# In[23]:


y_train.head()


# #### Building a Linear Model
# 
# We first need to import the `statsmodels.api` library using which we'll perform the linear regression.

# In[25]:


import statsmodels.api as sm


# By default, the `statsmodels` library fits a line on the dataset which passes through the origin. But in order to have an intercept, we need to manually use the `add_constant` attribute of `statsmodels`. And once we've added the constant to our `X_train` dataset, we can go ahead and fit a regression line using the `OLS` (Ordinary Least Squares) attribute of `statsmodels` as shown below

# In[27]:


# Add a constant to get an intercept
x_train_sm = sm.add_constant(x_train)

# Fit the resgression line using 'OLS'
lr = sm.OLS(y_train, x_train_sm).fit()


# In[28]:


# Print the parameters, i.e. the intercept and the slope of the regression line fitted
lr.params


# In[29]:


# Performing a summary operation lists out all the different parameters of the regression line fitted
print(lr.summary())


# ####  Looking at some key statistics from the summary
# The values we are concerned with are - 
# 1. The coefficients and significance (p-values)
# 2. R-squared
# 3. F statistic and its significance
# 
# ##### 1. The coefficient for TV is 0.054, with a very low p value
# The coefficient is statistically significant. So the association is not purely by chance. 
# 
# ##### 2. R - squared is 0.816
# Meaning that 81.6% of the variance in `Sales` is explained by `TV`
# 
# This is a decent R-squared value.
# 
# ###### 3. F statistic has a very low p value
# Meaning that the model fit is statistically significant, and the explained variance isn't purely by chance.
# 
# The fit is significant. Let's visualize how well the model fit the data.
# 
# From the parameters that we get, our linear regression equation becomes:
# 
# Sales = 6.948 + 0.054 * TV 

# In[31]:


plt.scatter(x_train,y_train)
plt.plot(x_train, 6.948 + 0.054*x_train, 'r')
plt.show()


# ## Step 4: Residual analysis 
# To validate assumptions of the model, and hence the reliability for inference
# 
# #### Distribution of the error terms
# We need to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.

# In[32]:


y_train_pred = lr.predict(x_train_sm)
res = (y_train - y_train_pred)


# In[33]:


fig = plt.figure()
sns.distplot(res, bins=15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()


# The residuals are following the normally distributed with a mean 0. All good!

# #### Looking for patterns in the residuals
# 

# In[35]:


plt.scatter(x_train,res)
plt.show()


# We are confident that the model fit isn't by chance, and has decent predictive power. The normality of residual terms allows some inference on the coefficients.
# 
# Although, the variance of residuals increasing with X indicates that there is significant variation that this model is unable to explain.

# As we can see, the regression line is a pretty good fit to the data

# ## Step 5: Predictions on the Test Set
# 
# Now that we have fitted a regression line on our train dataset, it's time to make some predictions on the test data. For this, we first need to add a constant to the `X_test` data like you did for `X_train` and then you can simply go on and predict the y values corresponding to `X_test` using the `predict` attribute of the fitted regression line.

# In[40]:


# Add a constant to x_test
x_test_sm = sm.add_constant(X_test)

# Predict the y values corresponding to x_test_sm
y_pred = lr.predict(x_test_sm)


# In[41]:


y_pred.head()


# In[48]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[43]:


# Checking the R-Squared on the test set
r_squared = r2_score(y_test, y_pred)
r_squared


# In[44]:


r_squared1 = r2_score(y_train, y_train_pred)
r_squared1


# In[50]:


# Checking the mean sqaured error for test set
print(mean_squared_error(y_test, y_pred))


# ##### Visualizing the fit on the test set

# In[52]:


plt.scatter(x_test, y_test)
plt.plot(x_test, 6.948 + 0.054 * x_test, 'r')
plt.show()


# ### Linear Regression using `linear_model` in `sklearn`

# In[53]:


# train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7, test_size=0.3, random_state=100)


# In[59]:


# Reshape x_train and x_test to (n,1)
x_train_lm = x_train.values.reshape(-1,1)
x_test_lm = x_test.values.reshape(-1,1)


# In[60]:


from sklearn.linear_model import LinearRegression

# Representing LinearRegression as lr
lm = LinearRegression()

# Fit the model
lm.fit(x_train_lm, y_train)


# In[64]:


# See the params, make predictions (train, test)
print(lm.coef_)
print(lm.intercept_)


# In[66]:


# Make predictions
y_train_pred = lm.predict(x_train_lm)
y_test_pred = lm.predict(x_test_lm)


# In[67]:


# Evaluate the model
print(r2_score(y_train, y_train_pred))
print(r2_score(y_test, y_test_pred))


# In[73]:


prediction=lm.predict([[180.8]])
prediction


# In[75]:


prediction1 = lm.predict([[44.5]])
prediction1

