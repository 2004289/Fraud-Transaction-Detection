# Fraud-Transaction-Detection

## Overview
This project uses machine learning to detect fraudulent financial transactions. The dataset is analyzed for patterns, preprocessed for model readiness, and several algorithms are trained and evaluated to predict whether a transaction is fraudulent or not.

## Objectives
Load and explore financial transaction data.

Handle missing values and outliers.

Identify class imbalance and perform resampling if needed.

Select important features.

Train machine learning models for classification.

Evaluate model accuracy using standard metrics.

## [1] Importing Required Libraries
````python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
````
## WRANGLING
### [2]: Load Dataset
````python
transaction_data = pd.read_csv("C:/Users/Asus/Downloads/Fraud (1).csv")
````
## [3]: Preview the entire dataset (you might want to limit this for performance)
````python
print(transaction_data)
````

## [4]: Shape of the dataset
````python
print("Dataset Shape:", transaction_data.shape)
````

## [5]: First 10 rows
````python
print(transaction_data.head(10))
````

## [6]: Last 10 rows
````python
print(transaction_data.tail(10))
````
## ANALYSING THE DATA
## [7]: Dataset information
````python
transaction_data.info()
````

## [8]: Check for missing values
````python
print("Any Null Values? ", transaction_data.isnull().values.any())
````

## [9]: Count of fraud and legit transactions
````python
print(transaction_data['isFraud'].value_counts())
````
## [10]: Calculate % of legit and fraud transactions
````python
legit = len(transaction_data[transaction_data.isFraud == 0])
fraud = len(transaction_data[transaction_data.isFraud == 1])
legit_transaction_percentage = (legit / (fraud + legit)) * 100
fraud_transaction_percentage = (fraud / (fraud + legit)) * 100

print("Number of Legit transactions: ", legit)
print("Number of Fraud transactions: ", fraud)
print("Percentage of Legit transactions: {:.4f} %".format(legit_transaction_percentage))
print("Percentage of Fraud transactions: {:.4f} %".format(fraud_transaction_percentage))
````
## DATA VISUALISATION
## [12]: Visualization of transaction classes
````python
plt.figure(figsize=(6, 6))
labels = ["Legit_Transactions", "Fraud_Transaction"]
count_classes = transaction_data.value_counts(transaction_data['isFraud'], sort=True)
count_classes.plot(kind="bar", rot=0)
plt.title("Visualization of Transactions")
plt.ylabel("Transaction Count")
plt.xticks(range(2), labels)
plt.show()
````
## new_dataset=transaction_data.copy()
````python
new_dataset=transaction_data.copy()
new_dataset.head()
````
````python
new_List = new_dataset.select_dtypes(include = "object").columns
print ("Variables with datatype - 'object' are:")
print (new_List)
````
````python
label_encode = LabelEncoder()
for i in new_List:
new_dataset[i] = label_encode.fit_transform(new_dataset[i].astype(str))
print (new_dataset.info())
````
## Multicolinearity Checking
````python
from statsmodels.stats.outliers_influence import variance_inflation_factor
 def calc_vif(transaction_data):
# Calculating Variance Inflation Factor
vif = pd.DataFrame()
vif["Variables"] = transaction_data.columns
vif["VIF"] = [variance_inflation_factor(transaction_data.values, i)
for i
in range(transaction_data.shape[1])]
return(vif)
calc_vif(new_dataset)
````
````python
new_dataset['balance_orig'] = new_dataset.apply(lambda x: x['oldbalanceOrg'] -␣
x['newbalanceOrig'],axis=1)
new_dataset['balance_dest'] = new_dataset.apply(lambda x: x['oldbalanceDest'] -␣
x['newbalanceDest'],axis=1)
new_dataset['name'] = new_dataset.apply(lambda x: x['nameOrig'] +␣
x['nameDest'],axis=1)
````
## dropping columns
````python
new_dataset = new_dataset.
drop(['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','nameOrig','nameDest'],calc_vif(new_dataset)
````
````python
corr=new_dataset.corr()
plt.figure(figsize=(6,6))
sns.heatmap(corr,annot=True)
````
## Model Building
````python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix,␣
ConfusionMatrixDisplay

X = new_dataset.drop(columns='isFraud', axis=1)
Y = new_dataset['isFraud']

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size= 0.3,␣
random_state= 42)
print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)
````
## Model Training
````python
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_dt = decision_tree.predict(X_test)
decision_tree_score = decision_tree.score(X_test, Y_test) * 100

random_forest = RandomForestClassifier(n_estimators= 100)
random_forest.fit(X_train, Y_train)
Y_pred_rf = random_forest.predict(X_test)
random_forest_score = random_forest.score(X_test, Y_test) * 100

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, Y_train)
Y_pred_lr = logistic_regression.predict(X_test)
logistic_regression_score = logistic_regression.score(X_test, Y_test) * 100

````

## Evaluation
````python
print("Decision Tree Score: ", decision_tree_score)
print("Random Forest Score: ", random_forest_score)
print("Logistic Regression Score: ", logistic_regression_score)

classification_report_dt = classification_report(Y_test, Y_pred_dt)
print("Classification Report for Decision Tree:")
print(classification_report_dt)
````
## Random Forest
````python
classification_report_rf = classification_report(Y_test, Y_pred_rf)
print("Classification Report for Random Forest:")
print(classification_report_rf)
````
## Logistic Regression
````python
classification_report_lr = classification_report(Y_test, Y_pred_lr)
print("Classification Report for Logistic Regression:")
print(classification_report_lr)
````

### CONCLUSION 
We can see the Accuracy of Decision Tree and Random Forest is almost same.
Precision is a crucial factor to predict correctly. The Precision and f1-score for Random Forest
is way better than other two. So, Random Forest is the best option. There is no way of taking
Logistic regression.
With the help of Correlation Heatmap, we have selected the variables.
Source of the transaction request,legitimacy of the requesting
organisation/individual could be the key factors to predict fraudulent customer.
verified software, usage of VPN, keeping contact with bank, keep updated
software on mobile/pc, using secure websites can prevent this kind of transactions




