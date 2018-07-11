
# coding: utf-8

# # Predicting Health/ Fitness goals using consumer wearable sensing devices
# 
# # by Devasena Inupakutika
# 

# In[474]:


#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from datetime import timedelta
import csv
import sys, os
from collections import deque


# ### Analysis of Fitbit Wearable Device Data written to csv file

# In[475]:


#Read the date-wise data from 5RJHGY_Fitbit.csv file
df = pd.read_csv('Fitbit-5RJHGY-data.csv')
pd.to_datetime(pd.Series(df['Date']), format="%Y-%m-%d")
df.head()


# We can see missing values as represented by NaN above. Hence, counting the number of missing values per column.

# ### Data Preprocessing

# In[476]:


df.isnull().sum()


# There are 317 and 399 i.e. all missing values in the sleep related columns of our dataset. Hence, filling with *zero* for missing values. 

# In[477]:


# fill missing values with mean column values
df.fillna(df.mean(), inplace=True)

# mark zero values as missing or NaN
df['Sleep Start Time'].fillna(0,inplace=True)
df['Sleep End Time'].fillna(0,inplace=True)
df['Minutes REM Sleep'].fillna(0,inplace=True)
df['Minutes Light Sleep'].fillna(0,inplace=True)
df['Minutes Deep Sleep'].fillna(0,inplace=True)

'''
df['Minutes Asleep'].fillna(0,inplace=True)
df['Minutes Awake'].fillna(0,inplace=True)
df['Number of Awakenings'].fillna(0,inplace=True)
df['Time in Bed'].fillna(0,inplace=True)
'''
#df[['Sleep Start Time','Sleep End Time','Minutes Asleep','Minutes Awake','Number of Awakenings','Time in Bed','Minutes REM Sleep','Minutes Light Sleep','Minutes Deep Sleep']] = df[['Sleep Start Time','Sleep End Time','Minutes Asleep','Minutes Awake','Number of Awakenings','Time in Bed','Minutes REM Sleep','Minutes Light Sleep','Minutes Deep Sleep']].replace(0, np.NaN)
df.head()
#df.isnull().sum()


# In[478]:


print("Dimensions of the data collected for last 1 year + starting from May 30, 2017: ",df.shape)


# ### Dropping the columns that are not required

# In[479]:


df.drop(df.columns[[3,9,10,15,16,17]], axis=1, inplace=True)


# In[480]:


df.head()


# In[481]:


print("Dimensions of the cleaned dataset: ",df.shape)


# ### Exploratory Data Analysis

# #### Feature distributions

# In[482]:


print(list(df))


# In[483]:


# Looking at the distributions corresponding to each numerical variable in the raw data
df.dtypes
h = df.hist(figsize = (15,20), layout = (6,3), xrot = 30)
plt.savefig('images/raw-data-eda.png', dpi=300)
plt.show()


# #### Initial Observations

# 1. Some of the data is zero: Reasons could be Fitbit is not worn, battery discharge or not synced for 10 consecutive days (In this case: it is mostly sleep data.)
# 2. Sedentary minutes are longer than activite minutes.
# 3. But majority calorie burn is due to activity calories which is some exercise or continuous walking or workout.
# 4. On average sleep is around 4-5 hours.
# 5. Daily steps vary between 5000 to 16000 which is close to 9-10 miles.
# 

# ### Looking at correlations

# #### Visualizing the important characteristics of a dataset: Observing pair-wise correlations between features

# Using the scatter plot below to see how the data is distributed and whether it has any outliers.

# In[484]:


#sns.set(style='whitegrid', context='notebook')
df_partial = df[['Steps','Distance','Minutes Sedentary','Calories Burned','Activity Calories','Minutes Asleep','Number of Awakenings',                     'Minutes Awake','Time in Bed']]
axes = pd.scatter_matrix(df_partial, figsize = (15,20), alpha=0.5, diagonal='kde')

corr = df_partial.corr().as_matrix()
for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes[i, j].annotate("%.3f" %corr[i,j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')
#sns.pairplot(df[cols], size=2.5)
plt.savefig('images/pairwise-correlation-matrix.png', dpi=300)
plt.show()


# ### Further Data Munging

# In[485]:


# Data cleaning and manipulation

# Create a weekday label which says which day of the week
df['weekday'] = df['Date'].map(lambda x: (datetime.strptime(str(x),"%Y-%m-%d")).weekday() , na_action = 'ignore')
df['day'] = df['Date'].map(lambda x: (datetime.strptime(str(x),"%Y-%m-%d")).date , na_action = 'ignore')
df['month'] = df['Date'].map(lambda x: (datetime.strptime(str(x),"%Y-%m-%d")).month , na_action = 'ignore')
# Percentage of awake time to time in bed (related to efficiency)
df['sleep_awake_per'] = df['Minutes Awake']/df['Time in Bed']*100 


# In[486]:


# Function to clean up plots
def prepare_plot_area(ax):
    # Remove plot frame lines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.spines["left"].set_visible(False) 
    
    # X and y ticks on bottom and left
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    
# Defining a color pattern that is pleasing
colrcode = [(31, 119, 180), (255, 127, 14),             (44, 160, 44), (214, 39, 40),              (148, 103, 189),  (140, 86, 75),              (227, 119, 194), (127, 127, 127),              (188, 189, 34), (23, 190, 207)]

for i in range(len(colrcode)):  
    r, g, b = colrcode[i]  
    colrcode[i] = (r / 255., g / 255., b / 255.)


# ### Data Interaction

# Here, we look at the trend shared by predictors, i.e the features that will be used to predict Steps. The correlation matrix is computed and represented as heatmap below:

# In[487]:


#Plotting correlation matrix as heatmap
cols = ['Calories Burned', 'Steps', 'Distance', 'Minutes Sedentary', 'Minutes Lightly Active', 'Minutes Fairly Active', 'Minutes Very Active', 'Activity Calories', 'Minutes Asleep', 'Minutes Awake', 'Number of Awakenings', 'Time in Bed']
cm = np.corrcoef(df[cols].values.T)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 5},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
plt.savefig('images/corr_heatmap.png', dpi=300)
plt.show()


# From above heatmap, we can observe some strong correlation between some sleep predictors.  Distance is strongly correlated to Steps and both are inter-correlated to Calories Burned and Activity Calories and also Minutes Very Active, which indicates that my main calorie burn is due to exercise or workout.

# ### Insights from Data Analysis

# My Fitbit Flex2 data shows some strong correlation between predictors such as Distance, Calories burned, Activity Calories and Minutes very active (1.00, 0.97, 0.96 and 0.88 correlation).  
# 

# #### Relation between Steps and Predictors are as shown in below graphs:

# ##### Steps Vs Predictors (Distance, Calories Burned and Activity Calories) All in One

# In[488]:


sns.pairplot(df, x_vars=['Distance','Calories Burned','Activity Calories'], y_vars='Steps', size=7, aspect=0.7, kind='reg')
plt.savefig('images/allinonestepsvspred.png', dpi=300)


# ##### Step Variations, Sleep Minutes and Sleep Inefficiency based on Week Days

# In[489]:


# Looking at variations based on weekday
steps_weekday = df['Steps'].groupby(df['weekday']).median()
sleep_minutes_asleep_med = df['Minutes Asleep'].groupby(df['weekday']).median()/60
sleep_eff = (1-df['Minutes Asleep']/df['Time in Bed'])*100
sl = sleep_eff.groupby(df['weekday']).median()


# In[490]:


# Median number of steps
fig,axes = plt.subplots(figsize=(12, 4), nrows=1, ncols=3)

ct = 0
plt.sca(axes[ct])
steps_weekday.plot(kind = 'bar',color = colrcode[0], alpha = 0.5)
plt.ylabel('Median number of steps')
plt.title('Daily Median number of steps walked')
plt.xticks(list(range(7)),['Mon','Tue','Wed','Thur','Fri','Sat','Sun'])
prepare_plot_area(axes[ct])

# Median number of minutes slept
ct +=1
plt.sca(axes[ct])
sleep_minutes_asleep_med.plot(kind = 'bar',color = colrcode[0], alpha = 0.5)
plt.ylabel('Median number of hours slept')
plt.title('Daily median number of hours slept')
plt.xticks(list(range(7)),['Mon','Tue','Wed','Thur','Fri','Sat','Sun'])
prepare_plot_area(axes[ct])

ct +=1
plt.sca(axes[ct])
sl.plot(kind = 'bar',color = colrcode[0], alpha = 0.5)
plt.ylabel('Median sleep inefficiency')
plt.title('Sleep Inefficiency %')
plt.xticks(list(range(7)),['Mon','Tue','Wed','Thur','Fri','Sat','Sun'])
prepare_plot_area(axes[ct])
plt.savefig('images/Steps-sleep-weekday.png', dpi=300)


# ##### Correlation between Step Count and Sleep Inefficiency

# In[491]:


fig = plt.figure(figsize = (12,6))
ax = fig.add_subplot(121)
ax.scatter(df['Steps'],df['sleep_awake_per'],color = colrcode[0])
plt.xlabel('Steps')
plt.ylabel('Awake time/total time in bed')
plt.title('Sleep inefficiency vs step count')
plt.savefig('images/sleepineff-steps.png', dpi=300)


# ### Steps Prediction and Evaluation

# Since the target variable is **Steps** here. In order to predict **Steps**, we split our data into train (70%) and test (30%) datasets.

# #### Simple Linear Regression Model

# In[492]:


from sklearn.cross_validation import train_test_split
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0)

#Starting it with one feature currently called 'Distance' because of correlation r =1 and estimating the coefficient
# create X and y
feature_cols = ['Distance']
X = df[feature_cols]
y = df.Steps

# instantiate and fit
slr = LinearRegression()
slr.fit(X, y)

# print the coefficients
print(slr.intercept_)
print(slr.coef_)

### STATSMODELS ###

# create a fitted model
lm1 = smf.ols(formula='Steps ~ Distance', data=df).fit()

# print the coefficients
lm1.params


# #### Using the model for prediction:
# y = 215.1933 + 2349.729 * x

# In[493]:


#For distance of 5 miles
slr.predict(5)


# In[494]:


### STATSMODELS ###

# We have to create a DataFrame since the Statsmodels formula interface expects it
X_new = pd.DataFrame({'Distance': [5]})

# predict for a new observation
lm1.predict(X_new)


# #### Plotting the least squares line

# In[495]:


def lin_regplot(X,y,model):
    plt.scatter(X,y,c="blue")
    plt.plot(X,model.predict(X),color="red")
    return None
    
lin_regplot(X,y,slr)
plt.xlabel("Distance in miles for the day")
plt.ylabel("Steps for the day")
plt.savefig('images/lin-reg.png', dpi=300)
plt.show()


# In[496]:


sns.pairplot(df, x_vars=['Distance','Calories Burned','Activity Calories'], y_vars='Steps', size=7, aspect=0.7, kind='reg')
plt.savefig('images/stepsvspred-model.png', dpi=300)


# #### Assessing variable importance using linear regression and test p-values for each predictor

# In[497]:


# print the p-values for the model coefficients for Distance Predictor
lm1.pvalues

#p-values for other variables and predictors

### STATSMODELS for Calories Burned###
#Removing space between columns
df=df.rename(columns={"Calories Burned":"Calories_Burned", "Activity Calories":"Activity_Calories","Minutes Very Active":"Minutes_Very_Active","Minutes Awake":"Minutes_Awake","Time in Bed":"Time_in_Bed","Minutes Asleep":"Minutes_Asleep"})


# In[498]:


df


# In[499]:


### STATSMODELS for calories_burned ###

# create a fitted model
lm11 = smf.ols(formula='Steps ~ Calories_Burned', data=df).fit()

# print the coefficients
lm11.params

### STATSMODELS ###

# We have to create a DataFrame since the Statsmodels formula interface expects it
X_new1 = pd.DataFrame({'Calories_Burned': [2000]})

# predict for a new observation
lm11.predict(X_new1)
# print the p-values for the model coefficients for Distance Predictor
lm11.pvalues


# In[500]:


### STATSMODELS for Activity_Calories ###

# create a fitted model
lm12 = smf.ols(formula='Steps ~ Activity_Calories', data=df).fit()

# print the coefficients
lm12.params

### STATSMODELS ###

# We have to create a DataFrame since the Statsmodels formula interface expects it
X_new2 = pd.DataFrame({'Activity_Calories': [1000]})

# predict for a new observation
lm12.predict(X_new2)
# print the p-values for the model coefficients for Calories Burned Predictor
lm12.pvalues


# In[501]:


### STATSMODELS for Minutes_Very_Active ###

# create a fitted model
lm13 = smf.ols(formula='Steps ~ Minutes_Very_Active', data=df).fit()

# print the coefficients
lm13.params

### STATSMODELS ###

# We have to create a DataFrame since the Statsmodels formula interface expects it
X_new3 = pd.DataFrame({'Minutes_Very_Active': [60]})

# predict for a new observation
lm13.predict(X_new3)
# print the p-values for the model coefficients for Minutes very active Predictor
lm13.pvalues


# In[502]:


### STATSMODELS for sleep_awake_per ###

# create a fitted model
lm14 = smf.ols(formula='Steps ~ sleep_awake_per', data=df).fit()

# print the coefficients
lm14.params

### STATSMODELS ###

# We have to create a DataFrame since the Statsmodels formula interface expects it
X_new4 = pd.DataFrame({'sleep_awake_per': [8.00]})

# predict for a new observation
lm14.predict(X_new4)
# print the p-values for the model coefficients for sleep_awake_per Predictor
lm14.pvalues


# #### How well model fits the data?

# In[503]:


slr.score(X,y)


# In[504]:


### STATSMODELS ###

# print the R-squared value for the model
lm1.rsquared


# #### Multiple Linear Regression

# In[505]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[506]:


# create X and y
feature_cols = ['Distance', 'Calories_Burned', 'Activity_Calories','sleep_awake_per']
X = df[feature_cols]
y = df.Steps

# instantiate and fit
slr2 = LinearRegression()
slr2.fit(X, y)

# print the coefficients
print(slr2.intercept_)
print(slr2.coef_)
# pair the feature names with the coefficients
list(zip(feature_cols, slr2.coef_))
slr2.score(X,y)


# In[507]:


lm14.summary()


# #### Evaluating Multiple Regression Model

# In[508]:


feature_cols = ['Distance', 'Calories_Burned', 'Activity_Calories','sleep_awake_per']
X = df[feature_cols]
y = df.Steps
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0)
# instantiate and fit
slr3 = LinearRegression()
slr3.fit(X_train, y_train)

y_train_pred = slr3.predict(X_train)
y_test_pred = slr3.predict(X_test)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# In[509]:


plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=0,xmax=30000, color='black', lw=2)
plt.tight_layout()

plt.savefig('images/eval_multiple_linear_regression_model-Residual-Plot.png', dpi=300)
plt.show()


# #### Using Regularized Methods for Regression (to tackle problems of Overfitting)

# #### Ridge Regression Model (L2 Penalized Model)

# In[510]:


from sklearn.linear_model import Ridge


# In[511]:


ridge=Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=0,xmax=30000, color='black', lw=2)
plt.tight_layout()

plt.savefig('images/Ridge-Residual-Plot.png', dpi=300)
plt.show()
slr3.score(X,y)


# In[512]:


ridge.score(X,y)


# #### LASSO Regression Model

# In[513]:


from sklearn.linear_model import Lasso
lasso=Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=0,xmax=30000, color='black', lw=2)
plt.tight_layout()

plt.savefig('images/Lasso-Residual-Plot.png', dpi=300)
plt.show()
lasso.score(X,y)


# #### Random Forest Regression

# In[514]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000,criterion="mse",random_state=1,n_jobs=-1)
forest.fit(X_train,y_train)
y_train_pred=forest.predict(X_train)
y_test_pred=forest.predict(X_test)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
plt.scatter(y_train_pred,  
            y_train_pred - y_train, 
            c='steelblue',
            edgecolor='white',
            marker='o', 
            s=35,
            alpha=0.9,
            label='training data')
plt.scatter(y_test_pred,  
            y_test_pred - y_test, 
            c='limegreen',
            edgecolor='white',
            marker='s', 
            s=35,
            alpha=0.9,
            label='test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=0, xmax=30000, lw=2, color='black')
plt.xlim([0, 30000])
plt.tight_layout()
plt.savefig('images/forest_regression_plot.png', dpi=300)
plt.show()

