import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Loading and Describing my dataset
data = pd.read_csv('Kc_house_data.csv')

#Setting format for better display
pd.options.display.float_format = '{:.2f}'.format

#Dropping non numeric columns for the summary
non_num_columns = ['id', 'date']
summary = data.drop(columns=non_num_columns).describe()  # displaying data information such as min value, max value, mean, standard deviation
print("Summary Statistics")
print(summary)

#Missing Values and Duplicates
print("\nBefore Cleaning \nMissing Cells :", data.isnull().sum().sum(), "and Duplicates: ", data.duplicated().sum())
#Cleaning my data
data.drop_duplicates(inplace=True) #Dropping duplicates
data.fillna(data.median(numeric_only=True), inplace=True) #replacing empty cells with median
print("\nAfter Cleaning \nMissing Cells: ", data.isnull().sum().sum(), " and Duplicates: ", data.duplicated().sum())

#Scaling features
#PURPOSE: Scaling ensures features contribute equally in regression. 

scaler = StandardScaler() #initializes the scaler
features = ['sqft_living', 'grade', 'bathrooms', 'lat', 'long', 'view', 'waterfront'] #added more features for modeling
data[features] = scaler.fit_transform(data[features])

#Splitting the dataset
# Log transform the target
y = np.log1p(data['price'])  #used the log1 so that i have clear belled graphs 
X = data[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  #splitting data into train and test datasets
print(f"\nSample Sizes for the split data. \nTrain: {X_train.shape[0]} \nTest: {X_test.shape[0]}")

#Distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True, color='green')
plt.title("Log Transformed Price Distribution", fontsize=15)
plt.xlabel("Price (logged price)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()


# PART B: Exploratory Data analysis
#Scatter Plots
#Top features influencing price:  sqft_living, grade and sqft_above 

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1) #1 row, 2 cloumns, 1st graph 
sns.scatterplot(data=data, x='sqft_living', y='price')
plt.title("Relationship between Living Area and Price")

plt.subplot(1, 2, 2)  #1 row, 2 cloumns, 2nd graph
sns.scatterplot(data=data, x='grade', y='price')
plt.title("Relationship between Grade and Price")
plt.show()

#Correlation Matrix
#STRONGEST PREDICTORS: sqft_living, grade and sqft_above 
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(numeric_only=True), annot=False, cmap='RdYlGn') #displays numerical data
plt.show()

#Detecting Outliers using Boxplot
# IMPACT ON OUTLIERS:
# Can distort regression models 
# Lead to poor predictions for normal houses 

sns.boxplot(x=data['price'])
plt.title("Price Outliers")
plt.show()


#PART C: Training the model
#Setting start time for my models
start = time.time()
lr = LinearRegression().fit(X_train, y_train) #trains the model
lr_pred = lr.predict(X_test)    #tests the model
end_lr = time.time() - start   #records end time

#printing out the prediction time
print(f"\nLinear Regression model trained in {end_lr:.4f} seconds")
#printing out the model predictions
print(f"\nLinear Regression model prediction {lr_pred}")

start = time.time()
dt = DecisionTreeRegressor(max_depth=8).fit(X_train, y_train)  #trains the model
dt_pred = dt.predict(X_test)     #tests the model
end_dt = time.time() - start    #records end time

#printing out the prediction time
print(f"\nDecision Tree Regression model trained in {end_dt:.4f} seconds")
#printing out the model predictions
print(f"\nDecision Tree Regression model prediction {dt_pred}")

#DIFFRERENCES: Decesion Tree regression is bit near to the actual value thatn Linear Regression 


#Plotting Predicted vs Actual for both models
# #Error patterns: Expensive houses tend to be underestimated 
# Outliers cause large prediction errors 
# Cross-validation is used to test model stability
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)  #1 row, 2 columns, 1st graph
plt.scatter(y_test, lr_pred, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Liner Regression: Actual vs Predicted")

plt.subplot(1, 2, 2)
plt.scatter(y_test, dt_pred, alpha=0.3, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Decision Tree: Actual vs Predicted")
plt.show()

#Storing predictions in a table
pred_table = pd.DataFrame({'Actual': y_test, 'Linear_Pred': lr_pred, 'Tree_Pred': dt_pred})
pd.options.display.float_format = '{:.2f}'.format
print("\nSample of Prediction Comparison Table:\n", pred_table.head())

#Comparing model training time and complexity
print(f"\nTraining Times:\nLinear Regression: {end_lr:.4f}s Low Complexity")
print(f"Decision Tree: {end_dt:.4f}s High Complexity since i used Depth of 8")

# Comparison table
#converting from logs to actual units
y_test_units = np.expm1(y_test)   # i had to change my values back to normal before the metric analysis
lr_pred_units = np.expm1(lr_pred)
dt_pred_units = np.expm1(dt_pred)

metrics = []
for machine, model in [("Linear Regression", lr_pred_units), ("Decision Tree", dt_pred_units)]:
    metrics.append({
        'Model': machine,
        'MAE': mean_absolute_error(y_test_units, model),
        'MSE': mean_squared_error(y_test_units, model),
        'R2': r2_score(y_test_units, model)
    })
pd.options.display.float_format = '{:.2f}'.format
print("\nModel Comparison Table\n", pd.DataFrame(metrics))

#Plot residuals for both models
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(y_test - lr_pred, kde=True, color='blue')
plt.title("Linear Regression Residuals Distribution")

plt.subplot(1, 2, 2)
sns.histplot(y_test - dt_pred, kde=True, color='green')
plt.title("Decision Tree Residuals Distribution")
plt.show()

# Target Transformation Validation:
#  The relatively symmetrical distribution of residuals around the red line confirms that the log transformation was correct. It prevented the outlier bullying that originally caused a poor 0.55 R2 score.
#Cross Validation for both models
for name, model in [("LR", lr), ("DT", dt)]:
    cross_validation_residue = cross_val_score(model, X, y, cv=5)
    print(f"{name}\n \nCross Validation R2 Score: {cross_validation_residue.mean():.4f} ± {cross_validation_residue.std():.4f}")


#Improvements: Added new extra features that would enable the model capture my data