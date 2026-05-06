import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,RocCurveDisplay)

#PART A: DATA PREPARATION

#Loading and describing structure
data = pd.read_csv('telco_customer_churn.csv')
print("Dataset Structure")
print(data.head(10))         # displaying top 10 rows of the dataset
print("\n", data.info())  # displaying data information such as data types and the space in memory

#Handling my missing values
print("\nHandling missing Values before and after")
print("\nBefore Handling")
print("\n",data.isnull().sum())  #displays the number of missing values per feature

print("\nAfter Handling")
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')  #converting the data in TotalCharges to numeric data and forces empty spaces to NaN
data.dropna(inplace=True) #dropping all empyt cells since now the contain Nan as their identifier
print("\n",data.isnull().sum())  

#Encoding categorical variables using One-Hot Encoding
print("\nEncoding Sample")
# Dropping customerID first as it is useless for training
data.drop('customerID', axis=1, inplace=True)

# Convert all object columns to dummy variables
data = pd.get_dummies(data, columns=data.select_dtypes(include=['object', 'str']).columns.drop('Churn')) #creates extra columns for columns with more than 1 unique alpha character
# Label Encoding only the target variable
le = LabelEncoder()    #since it only has 2 variables and being my target variable i couldnt split it into another column
data['Churn'] = le.fit_transform(data['Churn'])
print(data.head(4))  

#Scaling features
scaler = StandardScaler()     #initializing the scaler to handle my most differing columns in terms of range
scaled_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
data[scaled_columns] = scaler.fit_transform(data[scaled_columns])
print("\n Sample of the Scaled Data \n", data[scaled_columns].head(3))

#Splitting my dataset
X = data.drop(['Churn'], axis=1)  #dropping my target variable from the main dataset
y = data['Churn']  #assigning it to a new variable to keep track of it.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  #splitting data using the 80/20 rule
print(f"\n Number of Splitted datasets \n Train: {len(X_train)}, Test: {len(X_test)}")  #displaying number of rows of data in each category
print("\nSample for X train dataset ",X_train.head(3))
print("\nSample for y train dataset \n", y_train.head(3))


#PART B: Exploratory Data Analysis
#Scatter Plots
#HEre is used sublots for the scatter plots for easy comparisons
plt.figure(figsize=(20, 6))    #setting  the scatterplot window size
#Tenure vs Total Charges
plt.subplot(1, 2, 1)  #this shows that my graph has 1 row, 2 columns and the last (1) means that this particular graph is 1st in the row
sns.scatterplot(data=data, x='tenure', y='TotalCharges', hue='Churn', alpha=0.6) #the distribution is by churn
plt.title("Relationship between Tenure and Total Charges")
plt.xlabel("Tenure (Months)")
plt.ylabel("Total Charges")

#Monthly Charges vs Total Charges
plt.subplot(1, 2, 2) #1 row, 2 columns,  and its the 2nd graph
sns.scatterplot(data=data, x='MonthlyCharges', y='TotalCharges', hue='Churn', alpha=0.6) #the distribution is by churn
plt.title("Relationship between Monthly Charges and Total Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Total Charges")
plt.show()

#Correlation matrix
plt.figure(figsize=(10, 12)) #window size
sns.heatmap(data.select_dtypes(include=[np.number]).corr(), annot=False, cmap='RdYlGn') #selcting only datatypes that have numbers
plt.title("Correlation Heatmap")
plt.show()

#Boxplots
plt.figure(figsize=(14, 6))
#Tenure vs Churn
plt.subplot(1, 2, 1) #row, 2 columns, 1st graph
sns.boxplot(data=data, x='Churn', y='tenure', hue = 'Churn', palette='Set2', legend=False) #the color distribution is by churn
plt.title('Tenure Distribution by Churn')

#Monthly Charges vs Churn
plt.subplot(1, 2, 2) #1row, 2 columns, 2nd graph
sns.boxplot(data=data, x='Churn', y='MonthlyCharges', hue = 'Churn', palette='Set2', legend=False) #distibution by churn
plt.title('Monthly Charges Distribution by Churn')
plt.show()

#PART C:Training the model
#Initiated my models in a dictionary
models = {
    'LR': LogisticRegression(max_iter=1000),
    'kNN-3': KNeighborsClassifier(n_neighbors=3),
    'kNN-7': KNeighborsClassifier(n_neighbors=7),
    'DT': DecisionTreeClassifier(max_depth=5)
}

results = {} #created a dictionary for my results
training_time = {}  #created a dictionary for my model times

#Looping through the dictionary to train and test the models
for machine, model in models.items():
    starting_time = time.time()      #records start time
    model.fit(X_train, y_train)      #training my model
    stopping_time = time.time()      #recording end time
    
    #Storing test results and time
    results[machine] = model.predict(X_test)
    training_time[machine] = stopping_time - starting_time
    
    print(f"{machine} model trained in {training_time[machine]:.4f} seconds")

#comparing the models basing on time
time_comparison = pd.Series(training_time).sort_values() #sorting models according to time
print("\nComparing Models basing on Training time:")
print(time_comparison)

#Comparing models basing on prediction
print("\nComparing models basing on prediction")
pred_comparison = pd.DataFrame(results) #loads my results into dataframe or table 
print(pred_comparison.head())


#PART D: Ensemble Learning
#Hard voting
hv = VotingClassifier(estimators=[('lr', LogisticRegression(max_iter=1000)), ('knn', KNeighborsClassifier()), ('dt', DecisionTreeClassifier())], voting='hard').fit(X_train, y_train)
results['Hard Voting'] = hv.predict(X_test)  #increased iterations to 1000 to ensure convergence due to increased feature space from One-Hot Encoding

#Soft voting
sv = VotingClassifier(estimators=[('lr', LogisticRegression(max_iter=1000)), ('knn', KNeighborsClassifier()), ('dt', DecisionTreeClassifier())], voting='soft').fit(X_train, y_train)
results['Soft Voting'] = sv.predict(X_test)   

#Find number of differences
differ = np.sum(results['Hard Voting'] != results['Soft Voting']) #adding up rows where these models had different predictions
print(f"\nHard and Soft Voting disagreed on {differ} out of {len(y_test)} samples.")

#19. Stacking
stack = StackingClassifier(estimators=[('lr', LogisticRegression(max_iter=1000)), ('knn', KNeighborsClassifier())], final_estimator=LogisticRegression()).fit(X_train, y_train)
results['Stacking'] = stack.predict(X_test)   #combines the predictions of LR and KNN and runs them again in LR for more accurate results



#Creating a list to hold accuracy for all models
model_accuracy = []

#Adding accuracies from my models
for model in ['LR', 'kNN-3', 'kNN-7', 'DT']:
    accuracy = accuracy_score(y_test, results[model])
    model_accuracy.append({'Model Type': 'Individual', 'Model Name': model, 'Accuracy': accuracy}) #adding models to the list for every iteration

#Adding accuracies from my Ensemble models
for model in ['Hard Voting', 'Soft Voting', 'Stacking']:
    accuracy = accuracy_score(y_test, results[model])
    model_accuracy.append({'Model Type': 'Ensemble', 'Model Name': model, 'Accuracy': accuracy})

#Creating the table from my list
comparison_df = pd.DataFrame(model_accuracy)

print("\nComparison between Individual Models vs. Ensembles")
print(comparison_df.sort_values(by='Accuracy', ascending=False).to_string(index=False))



#PART E: Evaluation
#Performance Table
metrics = []  #created a list to store metrics
for model, pred in results.items():
    metrics.append({
        'Model': model,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1-score': f1_score(y_test, pred)
    })
print("\nEvaluation Table")
print(pd.DataFrame(metrics))

#Confusion Matrix
ConfusionMatrixDisplay.from_estimator(stack, X_test, y_test)
plt.title("Confusion matrix showing the TP,FP,TN and FN")
plt.xlabel("Predicted churns")
plt.xlabel("Actual churns")
plt.show()

#ROC curve
fig, ax = plt.subplots(figsize=(8, 6))
# Plotting each model directly onto the 'ax'
RocCurveDisplay.from_estimator(models['LR'], X_test, y_test, ax=ax, name='LR')
RocCurveDisplay.from_estimator(models['kNN-3'], X_test, y_test, ax=ax, name='k-NN')
RocCurveDisplay.from_estimator(models['kNN-7'], X_test, y_test, ax=ax, name='KNN-7')
RocCurveDisplay.from_estimator(models['DT'], X_test, y_test, ax=ax, name='Tree')
RocCurveDisplay.from_estimator(sv, X_test, y_test, ax=ax, name='Soft Voting')

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(alpha=0.3)
plt.show()
