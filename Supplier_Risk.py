# Kapil Sethi, Masters of Operational Research 2025, University of Delhi, India
# email id : kapil.du.or.25@gmail.com
# This code is a solution to the Supplier Risk Assessment Problem using Logistic Regression.

# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv("supplier_data.csv")
dataset = dataset.drop("Reputation", axis=1)
dataset = dataset.dropna()
dataset.info()

#Extracting the feature into X,
X = dataset.iloc[:,1:-1].values
#Extracting the target into y
y = dataset.iloc[:,-1].values
print(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Training the Logistic Regression model on the Training set
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy
accuracy = metrics.accuracy_score(y_test,y_pred)
print("Accuracy =", accuracy)

# printing classification report
print(metrics.classification_report(y_test, y_pred))

# New supplier data
risk_prediction_data = pd.read_csv("New_supplier_data.csv")
risk_prediction_data_features = risk_prediction_data.drop(['Supplier ID'], axis = 1)
risk_prediction_data_features = risk_prediction_data_features.drop("Reputation", axis=1)
risk_prediction_data_features = risk_prediction_data_features.dropna()
print(risk_prediction_data_features)

# Predicting probabilities
suppliers_risk_probs = classifier.predict_proba(risk_prediction_data_features)
print(suppliers_risk_probs)

# Displaying default probabilities
def sup_prob():
    suppliers_default_probs = suppliers_risk_probs[:, 1]
    print("Supplier's default probabilities :",suppliers_default_probs)
sup_prob()
